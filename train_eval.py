import os
import sys

sys.path.append(os.path.join('../'))
sys.path.append(os.path.join('../baseline/EV_GCN'))

import wandb
import pickle
import torch.nn as nn
from dataloader import Dataloader
from dpgc import DPGC
from lib.utils import *
from args import parse_args

root_folder = './dataset'
##################
# hyper-parameters
# lr = 0.01
# wd = 5e-5
# epochs = 300
# alpha = 0.1  # coefficient for site loss
# beta = 0.1  # coefficient for regularization of es
# gamma = 0.1  # coefficient for reconstruction error
args = parse_args()

if args.device == 0:
    device = torch.device("cuda:0")
else:
    device = torch.device("cuda:1")


print('  Loading dataset ...')
dl = Dataloader()
_, _, nonimg = dl.load_data()
# raw_features.shape = (871, 6105),  non-img.shape = (871, 3)

with open(f'{root_folder}/save/bas_cv', 'rb') as f:
    X, y, idxs_train, idxs_val, idxs_test = pickle.load(f)
    # X.shape = (871, 110, 110), y.shape = (871,), idxs_train: [[..], ..., [..]] (十折)

with open(f'{root_folder}/save/site_info_cv', 'rb') as f:
    y_site, idxs_train_site, idxs_val_site, idxs_test_site = pickle.load(f)

# overall results
corrects = np.zeros(10, dtype=np.int32)
accs = np.zeros(10, dtype=np.float32)
validation_accs = np.zeros(10, dtype=np.float32)
aucs = np.zeros(10, dtype=np.float32)
# folds_detailed_results = ""


# 10 折交叉验证
for fold in range(10):
    print("\r\n========================= Fold {} ============================".format(fold))
    train_ind = idxs_train[fold]
    val_ind = idxs_val[fold]
    test_ind = idxs_test[fold]

    # extract node features
    node_ftr = dl.get_node_features(train_ind)
    # node_ftr.shape = (871, 2000)
    # print(node_ftr.shape)

    # get PAE inputs
    edge_index, edgenet_input = dl.get_PEWE_inputs(nonimg)
    # edge_index.shape = (2, 18399),  edgenet_input.shape = (18399, 6)

    # normalization for PAE
    edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)

    # build network (using default parameters)
    model = DPGC(input_dim=2000, embedding_dim=2000, gcn_input_dim=2000, gcn_hidden_filters=16, gcn_layers=4, dropout=0.2, num_classes=2, non_image_dim=3, edge_dropout=0.5, config=args)
    model = model.to(device)

    # build loss, optimizer, metric
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # 学习率 指数衰减
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    features_cuda = torch.tensor(node_ftr, dtype=torch.float32).to(device)
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(device)
    labels = torch.tensor(y, dtype=torch.long).to(device)
    y_site = torch.tensor(y_site, dtype=torch.long).to(device)
    hyper_parameter_folder = f'./cv_models/{args.alpha}-{args.beta}-{args.epochs}-{args.gamma}-{args.lr}'
    if not os.path.exists(hyper_parameter_folder):
        os.makedirs(hyper_parameter_folder)
    fold_model_path = f'{hyper_parameter_folder}/fold-{fold}.pth'

    def train():
        print("  Number of training samples %d" % len(train_ind))
        print("  Start training...\r\n")
        best_acc = 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                label_logits, site_logits, es, reconstruct_features = model(features_cuda, edge_index, edgenet_input)
                # print(label_logits)
                # print(site_logits)
                loss_label = loss_fn(label_logits[train_ind], labels[train_ind])
                loss_site = loss_fn(site_logits, y_site)  # site labels
                reconstruct_errors = torch.norm(reconstruct_features - features_cuda)  # F-范数 or l1
                es_sparsity_regularization = torch.norm(es)  # F-范数 or l1
                loss = loss_label + args.alpha * loss_site + args.beta * es_sparsity_regularization + args.gamma * reconstruct_errors
                loss.backward()
                optimizer.step()  # 进行参数更新！！！
                lr_scheduler.step()  # 进行学习率更新！！（这个只能对学习率更新，并不能更新参数！！！）
            correct_train, label_acc_train = accuracy(label_logits[train_ind], labels[train_ind])
            _, site_acc_train = accuracy(site_logits, y_site)  #
            model.eval()
            with torch.set_grad_enabled(False):
                label_logits, site_logits, _, _ = model(features_cuda, edge_index, edgenet_input)
            label_loss_val = loss_fn(label_logits[val_ind], labels[val_ind])
            # site_loss_val = loss_fn(site_logits[val_ind], y_site[val_ind])
            correct_val, label_acc_val = accuracy(label_logits[val_ind], labels[val_ind])
            # _, site_acc_val = accuracy(site_logits[val_ind], y_site[val_ind])
            print("Epoch: {},\ttrain loss: {:.5f},\ttrain acc: {:.5f},\tval loss: {:.5f},\tval acc: {:.5f}".format(
                epoch, loss_label.item(), label_acc_train.item(), label_loss_val.item(), label_acc_val.item()))
            wandb.log({
                "Train Label Loss": loss_label.item(),
                "Train Site Loss": loss_site.item(),
                "Train Label Acc": label_acc_train,
                "Train Site ACC": site_acc_train,
                "Val Label Loss": label_loss_val.item(),
                # "Val Site Loss": site_loss_val.item(),
                "Val Label Acc": label_acc_val,
                # "Val Site Acc": site_acc_val,
            })
            if label_acc_val > best_acc:
                best_acc = label_acc_val
                torch.save(model.state_dict(), fold_model_path)
        validation_accs[fold] = best_acc
        wandb.log({
            "Fold Validation Accuracy": best_acc
        })
        print("  Fold {} best accuracy of val dataset: {:.5f}".format(fold, best_acc))

    def test():
        # global folds_detailed_results
        print("  Number of testing samples %d" % len(test_ind))
        print('  Start testing...')
        model.load_state_dict(torch.load(fold_model_path))
        model.eval()
        label_logits, site_logits, _, _ = model(features_cuda, edge_index, edgenet_input)
        corrects[fold], accs[fold] = accuracy(label_logits[test_ind], labels[test_ind])
        aucs[fold] = auc(label_logits[test_ind].detach().cpu().numpy(), labels[test_ind].detach().cpu().numpy())

        wandb.log({
            "Fold Test Accuracy": accs[fold]
        })
        print("  Fold {} test accuracy {:.5f}, test auc {:.5f}".format(fold, accs[fold], aucs[fold]))
        # folds_detailed_results += "  Fold {} test accuracy {:.5f}\n".format(fold, accs[fold])

    train()
    test()
print("\r\n========================== Finish ==========================")
n_samples = 0
for i in range(len(idxs_test)):
    n_samples += len(idxs_test[i])
acc_nfold = np.sum(corrects) / n_samples
print("=> Average test accuracy in {}-fold CV: {:.5f}".format(10, acc_nfold))
print("=> Average test auc in {}-fold CV: {:.5f}".format(10, np.mean(aucs)))
validation_acc_nfold = sum(validation_accs) / len(validation_accs)
wandb.log({
    "val_nfold": validation_acc_nfold,
    "acc_nfold": acc_nfold
})
# 保存结果到文件 --> 参数搜索结果记录.txt中
# with open('./参数搜索结果记录.txt', 'w') as f:
#     f.write(f"hyper-parameters: alpha={alpha}, beta={beta}, gamma={gamma} "
#             f"=> Average test accuracy {acc_nfold} \n")
#     f.write("Detailed results: \n")
#     f.write(folds_detailed_results)
#     f.write("\n\n\n\n")
# print("=> Average test AUC in {}-fold CV: {:.5f}".format(10, np.mean(aucs)))

