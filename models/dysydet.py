import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model

class MyModel(nn.Module):
    def __init__(self, clip_path, sd_path, num_classes, beta, len_t, device="cpu"):
        super(MyModel, self).__init__()

        self.beta = beta
        self.len_t = len_t
        self.device = device

        if clip_path is None or clip_path == "":
            clip_path = "openai/clip-vit-large-patch14"
        if sd_path is None or sd_path == "":
            sd_path = "runwayml/stable-diffusion-v1-5"

        self.clip_image_encoder = CLIPVisionModel.from_pretrained(clip_path)

        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet", revision=None,  use_safetensors=False)
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", revision=None)
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer", revision=None)
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", revision=None,  use_safetensors=False)
        self.noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
 
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 4 + self.len_t, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, num_classes)
        )

        self.oneDCNN = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                padding=1,
                bias=True
            ),
            nn.Tanh(),
            nn.Dropout(0.05)
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj"],
            init_lora_weights="loftq",
            loftq_config={
                "nbits": 4,
                "iterative": True,
                "reconstruction_error": 0.01
            },
            bias="none",
            inference_mode=False,
            use_dora=False,
            modules_to_save=None
        )

        self.clip_image_encoder = get_peft_model(self.clip_image_encoder, lora_config)

        for name, param in self.clip_image_encoder.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)


    def forward(self, image_tensor, clip_img_tensor, ensemble_size, t, prompt):

        clip_features = self.clip_image_encoder(pixel_values=clip_img_tensor, output_attentions=True)
        attentions = clip_features.attentions
        clip_feature = clip_features.pooler_output

        batch_size = clip_feature.shape[0]
        num_layers = len(attentions)
        sum_cls_attn = None
        beta = self.beta

        for i, attention in enumerate(attentions):
            cls_attn = attention.mean(dim=1)[:, 0, 1:]
            cls_attn_2d = cls_attn.view(batch_size, 1, 16, 16)
            if sum_cls_attn is None:
                sum_cls_attn = (1 - beta) * cls_attn_2d
            else:
                sum_cls_attn = beta * sum_cls_attn + (1 - beta) * cls_attn_2d

        avg_cls_attn_2d = sum_cls_attn / num_layers
        values, indices = torch.topk(avg_cls_attn_2d.flatten(1), k=100, dim=1)
        mask = torch.zeros_like(avg_cls_attn_2d.flatten(1))
        mask.scatter_(1, indices, 1.0)
        mask_2d_tensor = mask.view(batch_size, 1, 16, 16)       

        latents = self.vae.encode(image_tensor).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = latents.repeat((ensemble_size, 1, 1, 1))

        new_batch_size = latents.shape[0]
        text_input = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
                                return_tensors="pt"
                                ).to(self.device)
        encoder_hidden_states = self.text_encoder(text_input["input_ids"])[0]
        encoder_hidden_states = encoder_hidden_states.repeat((ensemble_size*batch_size, 1, 1))

        losses = []
        similarity_features = torch.zeros((batch_size, self.len_t), device=self.device)

        for idx, current_t in enumerate(t):
            noise = torch.randn_like(latents)
            timesteps = torch.randint(current_t, current_t + 1, (new_batch_size,),device=self.device)
            timesteps = timesteps.long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss_reshaped = loss.view(ensemble_size, batch_size, 4, 64, 64)
            average_loss = loss_reshaped.mean(dim=0)
            losses.append(average_loss)

            model_pred_reshaped = model_pred.view(ensemble_size, batch_size, -1)
            target_reshaped = target.view(ensemble_size, batch_size, -1)
            similarity = F.cosine_similarity(model_pred_reshaped, target_reshaped, dim=-1)
            average_similarity = similarity.mean(dim=0)
            similarity_features[:, idx] = average_similarity

        stacked_losses = torch.stack(losses, dim=0)
        avg_loss = stacked_losses.mean(dim=0)

        mask_up = F.interpolate(mask_2d_tensor.float(), size=(64, 64), mode='bilinear', align_corners=False)
        masked_features = avg_loss * mask_up
        mean_features = masked_features.mean(dim=[2, 3])
        similarity_features = similarity_features.unsqueeze(1)
        similarity_features = self.oneDCNN(similarity_features)
        similarity_features = similarity_features.squeeze(1)


        feature = torch.cat((clip_feature, mean_features, similarity_features), dim=1)
        output = self.classifier(feature)
        return output