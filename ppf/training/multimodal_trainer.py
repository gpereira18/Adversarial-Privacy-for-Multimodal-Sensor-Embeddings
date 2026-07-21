import torch.nn as nn
import torch
import torch.nn.functional as F
from ppf.models.grl import grad_reverse
from ppf.training.schedule import lambda_schedule, lr_schedule

class MultimodalTrainer:
    def __init__(
        self, 
        encoders: dict, 
        projectors: dict, 
        task_head, probe, 
        device="mps",
        lr_encoder=1e-4,
        lr_probe=1e-3,
        lr_projector=1e-3,
        current_lambda=0.0,
        lambda_max=1.0,
        total_epochs=15,
        warmup_epochs=5,
        
        ):
        
        self.encoders = encoders
        self.projectors = projectors
        self.task_head = task_head
        self.probe = probe

        enc_params = list(self.task_head.parameters())
        for enc in self.encoders.values():
            enc_params += list(enc.parameters())
        for proj in self.projectors.values():
            enc_params += list(proj.parameters())
        self.opt_encoder = torch.optim.Adam(enc_params, lr=lr_encoder)
        self.opt_probe = torch.optim.Adam(self.probe.parameters(), lr=lr_probe)
        self.lr_encoder = lr_encoder
        self.lr_probe = lr_probe
        self.lr_projector = lr_projector
        self.current_lambda = current_lambda
        self.lambda_max = lambda_max
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

        self.device = torch.device(device)
        self.task_head.to(self.device)
        self.probe.to(self.device)
        for mod in self.encoders.values():
            mod.to(self.device)
        for mod in self.projectors.values():
            mod.to(self.device)

    def _compute_embeddings(self,batch):
        all_idx = []
        all_emb = []
        for modality in self.encoders.keys():
            idx = [i for i, m in enumerate(batch["modality"]) if m == modality]
            if idx:
                inputs = torch.stack([batch["input_tensor"][i] for i in idx]).to(self.device)
                emb = self.projectors[modality](self.encoders[modality](inputs))
                all_idx.extend(idx)
                all_emb.append(emb)
        cat = torch.cat(all_emb, dim=0)
        order = torch.argsort(torch.tensor(all_idx, device=self.device))
        return cat[order]

    def train_epoch(self, train_loader, epoch):
        self._update_schedule(epoch)
        total_task_loss, total_probe_loss, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            activity = batch["activity"].to(self.device)
            subject = batch["subject"].to(self.device)

            with torch.no_grad():
                emb_frozen = self._compute_embeddings(batch)

            for _ in range(5):
                self.opt_probe.zero_grad()
                probe_logits = self.probe(emb_frozen)
                probe_loss = F.cross_entropy(probe_logits, subject)
                probe_loss.backward()
                self.opt_probe.step()

            emb = self._compute_embeddings(batch)
            
            self.opt_encoder.zero_grad()
            self.opt_probe.zero_grad()
            task_logits = self.task_head(emb)
            probe_logits = self.probe(grad_reverse(emb, self.current_lambda))
            loss = F.cross_entropy(task_logits, activity) + F.cross_entropy(probe_logits, subject)
            loss.backward()
            self.opt_encoder.step()
            self.opt_probe.step()

            total_task_loss += F.cross_entropy(task_logits, activity).item()
            total_probe_loss += F.cross_entropy(probe_logits, subject).item()
            n_batches += 1

        return {
            "task_loss": total_task_loss / n_batches,
            "probe_loss": total_probe_loss / n_batches,
            "lambda": self.current_lambda,
            }

    def _update_schedule(self, epoch):
        if epoch <= self.warmup_epochs:
            self.current_lambda = 0.0
        else:
            adv_epoch = epoch - self.warmup_epochs
            adv_total = self.total_epochs - self.warmup_epochs
            self.current_lambda = lambda_schedule(
                adv_epoch, adv_total, self.lambda_max
            )

        new_lr_enc = lr_schedule(epoch, self.total_epochs, self.lr_encoder)
        for pg in self.opt_encoder.param_groups:
            pg["lr"] = new_lr_enc
            
    def evaluate(self, task_loader, probe_loader):
        self.task_head.eval()
        self.probe.eval()
        for mod in self.encoders.values():
            mod.eval()
        for mod in self.projectors.values():
            mod.eval()

        with torch.no_grad():
            task_correct, total_task = 0, 0
            for batch in task_loader:
                activity = batch["activity"].to(self.device)
                emb = self._compute_embeddings(batch)
                task_preds = self.task_head(emb).argmax(dim=1)
                task_correct += (task_preds == activity).sum().item()
                total_task += len(activity)
            
            probe_correct, total_probe = 0, 0
            for batch in probe_loader:
                subject = batch["subject"].to(self.device)
                emb = self._compute_embeddings(batch)
                probe_preds = self.probe(emb).argmax(dim=1)
                probe_correct += (probe_preds == subject).sum().item()
                total_probe += len(subject)

        self.task_head.train()
        self.probe.train()
        for mod in self.encoders.values():
            mod.train()
        for mod in self.projectors.values():
            mod.train()

        return {
            "task_accuracy":  task_correct  / total_task,
            "probe_accuracy": probe_correct / total_probe,
        }
            

           
            