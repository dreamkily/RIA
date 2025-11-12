import torch
import time
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model import Encoder_overall
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from .preprocess import adjacent_matrix_preprocessing
from .optimal_clustering_HLN import R5


class Train:
    def __init__(self,
                 data,
                 datatype,
                 device,
                 random_seed=2024,
                 dim_input=3000,
                 dim_output=64,
                 Arg=None,
                 log_dir="logs"
                 ):

        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to_dense().to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to_dense().to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to_dense().to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to_dense().to(self.device)

        self.paramed_adj_omics1 = Parametered_Graph(self.adj_feature_omics1, self.device).to(self.device)
        self.paramed_adj_omics2 = Parametered_Graph(self.adj_feature_omics2, self.device).to(self.device)

        self.adj_feature_omics1_copy = copy.deepcopy(self.adj_feature_omics1)
        self.adj_feature_omics2_copy = copy.deepcopy(self.adj_feature_omics2)

        self.EMA_coeffi = 0.9
        self.K = 5
        self.T = 4
        self.arg = Arg

        self.clustering = R5(self.datatype, self.arg)

        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs

        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        if self.datatype == 'SPOTS':
            self.epochs = 200
            self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight]
            self.weight_decay = 5e-3
            self.learning_rate = 0.001

        elif self.datatype == 'Stereo-CITE-seq':
            self.epochs = 300
            self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight]
            self.weight_decay = 5e-2
            self.learning_rate = 0.001

        elif self.datatype == '10x':
            self.learning_rate = 0.01
            self.epochs = 30
            self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight]
            self.weight_decay = 5e-3
            self.EMA_coeffi = Arg.alpha

        elif self.datatype == 'Spatial-epigenome-transcriptome':
            self.epochs = 300
            self.weight_factors = [Arg.RNA_weight, Arg.ADT_weight]
            self.learning_rate = 0.0001
            self.weight_decay = 5e-2

        model_params = {
            "datatype": self.datatype,
            "random_seed": self.random_seed,
            "dim_input": self.dim_input,
            "dim_output": self.dim_output,
            "dim_input1": self.dim_input1,
            "dim_input2": self.dim_input2,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "weight_factors": self.weight_factors,
            "EMA_coefficient": self.EMA_coeffi
        }
        self.logger.log_model_params(model_params)

        data_info = {
            "n_cells_omics1": self.n_cell_omics1,
            "n_cells_omics2": self.n_cell_omics2,
            "features_omics1_shape": self.features_omics1.shape,
            "features_omics2_shape": self.features_omics2.shape
        }
        self.logger.log_data_info(data_info)

    def train(self, alpha=1.0, beta=1.0):
        """
        Trains the model.

        Args:
            alpha (float): Weight for the Lbnm loss term.
            beta (float): Weight for the Ldpcl (clustering) loss term.
        """
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(
            self.device)
        self.optimizer = torch.optim.SGD(list(self.model.parameters()) +
                                         list(self.paramed_adj_omics1.parameters()) +
                                         list(self.paramed_adj_omics2.parameters()),
                                         lr=self.learning_rate,
                                         momentum=0.9,
                                         weight_decay=self.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        self.model.train()
        from torch.cuda.amp import GradScaler, autocast
        for epoch in tqdm(range(self.epochs), desc=f"Training (alpha={alpha}, beta={beta})"):
            scaler = torch.cuda.amp.GradScaler()

            with autocast():
                self.model.train()

                results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                     self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)

                matrix1 = results['emb_latent_omics1']
                matrix1 = matrix1.to(torch.float32)
                _, s1, _ = torch.svd(matrix1)
                loss_bnm1 = - torch.sum(s1) / matrix1.size(0)

                matrix2 = results['emb_latent_omics2']
                matrix2 = matrix2.to(torch.float32)
                _, s2, _ = torch.svd(matrix2)
                loss_bnm2 = - torch.sum(s2) / matrix2.size(0)

                loss_bnm = self.weight_factors[0] * loss_bnm1 + self.weight_factors[1] * loss_bnm2

                updated_adj_omics1 = self.paramed_adj_omics1()
                updated_adj_omics2 = self.paramed_adj_omics2()

                loss_fro = (torch.norm(updated_adj_omics1 - self.adj_feature_omics1_copy.detach(), p='fro') +
                            torch.norm(updated_adj_omics2 - self.adj_feature_omics2_copy.detach(), p='fro')) / 2

                clustering_loss = self.clustering(results['emb_latent_combined'], epoch)

                # ==================== FIX for AttributeError ====================
                # Ensure clustering_loss is a tensor. The self.clustering method
                # might return a standard Python float, which does not have the
                # .item() method. This converts it to a tensor if needed.
                if not isinstance(clustering_loss, torch.Tensor):
                    clustering_loss_tensor = torch.tensor(clustering_loss, device=self.device)
                else:
                    clustering_loss_tensor = clustering_loss
                # =============================================================

                # Use the passed alpha and beta to calculate the final loss
                loss = alpha * loss_bnm + beta * clustering_loss_tensor + loss_fro

                if epoch % 10 == 0:
                    print(
                        f"\n[Epoch {epoch}] "
                        f"BNM Loss: {loss_bnm.item():.4f} | "
                        f"Loss Fro: {loss_fro.item():.4f} | "
                        f"Clustering Loss: {clustering_loss_tensor.item():.4f} | "
                        f"Total Loss: {loss.item():.4f}"
                    )

                self.logger.log_epoch(
                    epoch=epoch,
                    bnm_loss=loss_bnm,
                    loss_h=loss_fro,
                    clustering_loss=clustering_loss_tensor,
                    total_loss=loss
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.adj_feature_omics1 = self.paramed_adj_omics1()
            self.adj_feature_omics2 = self.paramed_adj_omics2()

            self.adj_feature_omics1_copy = self.EMA_coeffi * self.adj_feature_omics1_copy + (
                    1 - self.EMA_coeffi) * self.adj_feature_omics1.detach().clone()
            self.adj_feature_omics2_copy = self.EMA_coeffi * self.adj_feature_omics2_copy + (
                    1 - self.EMA_coeffi) * self.adj_feature_omics2.detach().clone()

        print("Model training finished!\n")

        start_time = time.time()

        with torch.no_grad():
            self.model.eval()
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2)

        end_time = time.time()
        infer_time = end_time - start_time
        print("Infer time: ", end_time - start_time)

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        A_no_diag = self.paramed_adj_omics2().cpu().detach().clone()
        A_no_diag.fill_diagonal_(0)

        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'RIA': emb_combined.detach().cpu().numpy(),
                  'adj_feature_omics1': self.adj_feature_omics1.detach().cpu().numpy()
                  }

        final_results = {
            "inference_time": infer_time,
            "emb_latent_omics1_shape": output['emb_latent_omics1'].shape,
            "emb_latent_omics2_shape": output['emb_latent_omics2'].shape,
            "RIA_shape": output['RIA'].shape
        }
        self.logger.log_final_results(final_results)

        embeddings = {
            "RIA": output['RIA'],
            "emb_latent_omics1": output['emb_latent_omics1'],
            "emb_latent_omics2": output['emb_latent_omics2']
        }
        self.logger.save_embedding_plots(embeddings)

        self.logger.close()

        return output


class Parametered_Graph(nn.Module):
    def __init__(self, adj, device):
        super(Parametered_Graph, self).__init__()
        self.adj = adj
        self.device = device

        n = self.adj.shape[0]
        self.paramed_adj_omics = nn.Parameter(torch.FloatTensor(n, n))
        self.paramed_adj_omics.data.copy_(self.adj)

    def forward(self, A=None):
        if A is None:
            adj = (self.paramed_adj_omics + self.paramed_adj_omics.t()) / 2
        else:
            adj = (A + A.t()) / 2

        adj = nn.ReLU(inplace=True)(adj)
        normalized_adj = self._normalize(adj.to(self.device) + torch.eye(adj.shape[0]).to(self.device))
        return normalized_adj.to(self.device)

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx
