import torch as T
import torch.nn as nn
import torch.optim as opt
from src.pecan_dataport.participant_preprocessing import PecanParticipantPreProcessing
from src.dataset import PecanDataModule
from src.models.gru import GRUModel
from src.models.lstm import LSTMModel
from src.models.rnn import RNNModel
from src.models.recorrent_ensemble import RecorrentEnsemble


class Engine:
    def __init__(self, ensemble_array: nn.ModuleList, device):
        models_name = [type(model).__name__ for model in ensemble_array]
        optimizers = [opt.AdamW(model.parameters(), lr=1e-5) for model in ensemble_array]

        self.trainer = []
        for model_name, model_class, model_opt in zip(models_name, ensemble_array, optimizers):
            self.trainer.append({
                'model_name': model_name,
                'model_class': model_class,
                'model_opt': model_opt
            })
        self.device = device

    @staticmethod
    def _loss_fn(output, label):
        return nn.MSELoss()(output, label)

    def train(self, dataloader, epoch, num_epochs):
        #treinar os modelos separadamente
        final_losses = {model['model_name']: 0 for model in self.trainer}
        for i, data in enumerate(dataloader):
            for model in self.trainer:
                loss = 0
                model['model_class'].train()
                model['model_opt'].zero_grad()

                sequence = data['sequence'].to(self.device)
                label = data['label'].to(self.device)
                output = model['model_class'](sequence)
                train_loss = self._loss_fn(output, label.unsqueeze(dim=1))
                train_loss.backward()
                loss += train_loss.item()
                final_losses[model['model_name']] += loss / len(dataloader)
                break
            if i % 10 == 0:
                print(f"[!] - [{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss:{final_losses}")
            break

        return final_losses


    def eval(self, dataloader, epoch, num_epochs):
        #eval os modelos e treinar o ensemble

        #treinar pela validação o ensemble
        final_losses = {model['model_name']: 0 for model in self.trainer}
        for i, data in enumerate(dataloader):
            for model in self.trainer:
                model['model_class'].eval()
                loss = 0
                sequence = data['sequence'].to(self.device)
                label = data['label'].to(self.device)

                output = model['model_class'](sequence)
                eval_loss = self._loss_fn(output, label.unsqueeze(dim=1))
                loss += eval_loss.item()

                final_losses[model['model_name']] += loss / len(dataloader)
                break
            break

        ensemble_model = RecorrentEnsemble(self.device, models=self.trainer)
        ensemble_model.train()
        loss = 0
        for i, data in enumerate(dataloader):
            ensemble_model.zero_grad()
            sequence = data['sequence'].to(self.device)
            label = data['label'].to(self.device)
            output = ensemble_model(sequence, label)
            train_loss = self._loss_fn(output, label.unsqueeze(dim=1))
            train_loss.backward()
            loss += train_loss.item()
        final_losses['ensemble'] = loss/len(dataloader)

        return final_losses

def main():
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    pecan_dataset = PecanParticipantPreProcessing('661_test_30_pca', 'data/participants_data/1min/', 60, task='train')

    train_sequences, test_sequences, val_sequences = pecan_dataset.get_sequences()
    n_features = pecan_dataset.get_n_features()

    data_module = PecanDataModule(
        device=device,
        train_sequences=train_sequences,
        test_sequences=test_sequences,
        val_sequences=val_sequences,
        batch_size=32,
        num_workers=0,
        pin_memory=True
    )
    data_module.setup()

    gru_model = GRUModel(device=device, input_dim=n_features, hidden_dim=256, layer_dim=3, dropout_prob=0.3, activation_function='sigmoid')
    rnn_model= RNNModel(device=device, n_features=n_features, n_hidden=256, n_layers=3, drop_out=0.3, activation_function='sigmoid')
    lstm_model= LSTMModel(device=device, n_features=n_features, n_hidden=256, n_layers=3, dropout=0.3, activation_function='sigmoid', bidirectional=False)

    list_models = nn.ModuleList([gru_model, rnn_model, lstm_model])
    engine = Engine(ensemble_array=list_models, device=device)
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss = engine.train(data_module.train_dataloader(), epoch, num_epochs)
        eval_loss = engine.eval(data_module.val_dataloader(), epoch, num_epochs)
        print(f"[{epoch}/{num_epochs}]\tTrain_Loss:{train_loss}\tValidation_Loss:{eval_loss}")
        break


if __name__ == "__main__":
    main()