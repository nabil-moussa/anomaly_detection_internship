import torch
import torch.nn as nn
from modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel,
    VAEReconstructionModel,
)


class MTAD_GAT(nn.Module):
    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=False,
        gru_n_layers=1,
        gru_hid_dim=300,
        forecast_n_layers=3,
        forecast_hid_dim=300,
        recon_n_layers=1,
        recon_hid_dim=300,
        recon_out_dim=None,
        dropout=0.3,
        alpha=0.2,
        use_vae=False,
    ):
        super(MTAD_GAT, self).__init__()
        self.use_vae = use_vae

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(
            n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2
        )
        self.temporal_gat = TemporalAttentionLayer(
            n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2
        )
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(
            gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout
        )

        recon_out_dim = recon_out_dim or out_dim
        if use_vae:
            self.recon_model = VAEReconstructionModel(
                window_size, gru_hid_dim, recon_hid_dim, recon_out_dim, recon_n_layers, dropout
            )
        else:
            self.recon_model = ReconstructionModel(
                window_size, gru_hid_dim, recon_hid_dim, recon_out_dim, recon_n_layers, dropout
            )

    def forward(self, x):
        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        h_cat  = torch.cat([x, h_feat, h_temp], dim=2)
        _, h_end = self.gru(h_cat)
        h_end  = h_end.view(x.shape[0], -1)
    
        predictions = self.forecasting_model(h_end)
    
        if self.use_vae:
            recon_mu, recon_log_sigma, kl_loss = self.recon_model(h_end)
            return predictions, recon_mu, recon_log_sigma, kl_loss
        else:
            recons = self.recon_model(h_end)
            return predictions, recons