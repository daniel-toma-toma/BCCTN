import torch

from DCNN.utils.apply_mask import apply_mask

from .model import DCNN




class BinauralAttentionDCNN(DCNN):

    def forward(self, inputs):
        # batch_size, binaural_channels, time_bins = inputs.shape
        cspecs_l = self.stft(inputs[:, 0])
        cspecs_r = self.stft(inputs[:, 1])
        cspecs = torch.stack((cspecs_l, cspecs_r), dim=1)
        encoder_out_l = self.encoder(cspecs_l.unsqueeze(1))
        encoder_out_r = self.encoder(cspecs_r.unsqueeze(1))
        
        # 2. Apply attention
        if not self.symmetric:
          attention_in = torch.cat((encoder_out_l[-1],encoder_out_r[-1]), dim=1)
          x_attn = self.mattn(attention_in)
          x_l_mattn = x_attn[:,:128,:,:]
          x_r_mattn = x_attn[:,128:,:,:]
        else:
          attention_in1 = torch.cat((encoder_out_l[-1],encoder_out_r[-1]), dim=1)
          attention_in2 = torch.cat((encoder_out_r[-1],encoder_out_l[-1]), dim=1)
          x_attn1 = self.mattn(attention_in1)
          x_attn2 = self.mattn(attention_in2)
          x_l_mattn1 = x_attn1[:,:128,:,:]
          x_r_mattn1 = x_attn1[:,128:,:,:]
          x_l_mattn2 = x_attn2[:,128:,:,:]
          x_r_mattn2 = x_attn2[:,:128,:,:]
          x_l_mattn = (x_l_mattn1 + x_l_mattn2) / 2
          x_r_mattn = (x_r_mattn1 + x_r_mattn2) / 2
        
        # 3. Apply decoder
        x_l = self.decoder(x_l_mattn, encoder_out_l)
        x_r = self.decoder(x_r_mattn, encoder_out_r)

        # 4. Apply mask
        out_spec_l = apply_mask(x_l[:, 0], cspecs_l, self.masking_mode)
        out_spec_r = apply_mask(x_r[:, 0], cspecs_r, self.masking_mode)

        # 5. Invert STFT
        out_wav_l = self.istft(out_spec_l)
        out_wav_r = self.istft(out_spec_r)
        out_wav = torch.stack([out_wav_l, out_wav_r], dim=1)
        #print(f"  cspecs_l: {cspecs_l.shape}, cspecs: {cspecs.shape}, x_l_mattn: {x_l_mattn.shape}, x_l: {x_l.shape},\\
        #          out_spec_l: {out_spec_l.shape}, out_wav_l: {out_wav_l.shape}, out_wav: {out_wav.shape}")
       
        return out_wav
