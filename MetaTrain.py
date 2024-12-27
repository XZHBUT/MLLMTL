import math
from copy import deepcopy

import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from Model.CNN import CNNModel_Mate
from Tool.Build_Task import Build_Task_from_HUST_JNU_HIT, Build_Task_from_CWRU_All, Build_Task_from_HUST_JNU_CWRU, \
    Build_Task_from_HIT_All, Build_Task_from_HUST_HIT_CWRU, Build_Task_from_JNU_All, Build_Task_from_SEU_All, \
    Build_Task_from_JNU_HIT_CWRU, Build_Task_from_HUST_SelectWay
from Tool.MetaTrainFunction import MaML_Finetunning_with_Learnable_triplet, MAMl_train_with_Learnable_triplet

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = CNNModel_Mate(num_classes=5, drouput=0.5, initial_margin=initial_margin)
    # model.feature_extractor.load_state_dict(torch.load("CNNModel__HUST_JNU_HIT.pth"))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=max_lr)

    Train_Task_s_x, Train_Task_s_y, Train_Task_q_x, Train_Task_q_y = Build_Task_from_JNU_HIT_CWRU(
        filepath='data', Task=100, SampleLength=2048, step=1024, Way=3, Shot=5,
        QueryShot=5)

    Test_Task_s_x, Test_Task_s_y, Test_Task_q_x, Test_Task_q_y = Build_Task_from_HUST_SelectWay(
        filepath='data/HUST', Task=10, Select_Class=[9, 10, 6],
        SampleLength=2048, Way=3, Shot=5, QueryShot=5)

    Train_dataset = TensorDataset(Train_Task_s_x, Train_Task_s_y, Train_Task_q_x, Train_Task_q_y)
    Test_dataset = TensorDataset(Test_Task_s_x, Test_Task_s_y, Test_Task_q_x, Test_Task_q_y)

    # train_loader = DataLoader(Train_dataset, batch_size=Train_bz, shuffle=True)
    # test_loader = DataLoader(Test_dataset, batch_size=Test_bz, shuffle=True)

    curr_test_loss = 9999.0
    for epoch in range(num_epochs):
        train_loader = DataLoader(Train_dataset, batch_size=Train_bz, shuffle=True)

        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(epoch / num_epochs * math.pi))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()

        for i, (bz_s_x, bz_s_y, bz_q_x, bz_q_y) in enumerate(train_loader):
            meta_loss, meta_acc = MAMl_train_with_Learnable_triplet(model=model, support_images=bz_s_x.to(device),
                                                                    support_labels=bz_s_y.to(device),
                                                                    query_images=bz_q_x.to(device),
                                                                    query_labels=bz_q_y.to(device),
                                                                    inner_step=inner_step,
                                                                    inner_lr=inner_lr,
                                                                    optimizer=optimizer,
                                                                    n_classes=10, alpha=0.5, PK=PK, NK=NK,
                                                                    is_train=True)
            print("epoch:" + str(epoch + 1) + '-----'
                  + f'current train task query loss: {meta_loss.item():.4f}' + '-----'
                  + f'current train task query acc: {meta_acc:.4f}')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            test_loader = DataLoader(Test_dataset, batch_size=Test_bz, shuffle=True)


            initial_state_dict = deepcopy(model.state_dict())

            print("===================================Test Task start===================================")
            print("epoch:" + str(epoch + 1) + '-----' + "Testing the effects of fine-tuning")
            All_test_acc = 0.0
            All_test_loss = 0.0
            for j, (bz_t_s_x, bz_t_s_y, bz_t_q_x, bz_t_q_y) in enumerate(test_loader):
                meta_test_loss, meta_test_acc, meta_acc_list = MaML_Finetunning_with_Learnable_triplet(model=model,
                                                                                                       support_images=bz_t_s_x.to(
                                                                                                           device),
                                                                                                       support_labels=bz_t_s_y.to(
                                                                                                           device),
                                                                                                       query_images=bz_t_q_x.to(
                                                                                                           device),
                                                                                                       query_labels=bz_t_q_y.to(
                                                                                                           device),
                                                                                                       inner_step=inner_step,
                                                                                                       alpha=0.5, PK=PK,
                                                                                                       NK=NK,
                                                                                                       inner_lr=inner_lr)
                All_test_loss += meta_test_loss.item()
                All_test_acc += meta_test_acc * len(bz_t_q_x)
                print("epoch:" + str(epoch + 1) + '-----'
                      + f'current test task query loss : {meta_test_loss.item():.4f}' + '-----'
                      + f'current test task query acc: {meta_test_acc:.4f}')
                print(f"Test_acc_list:")
                print(meta_acc_list)
            Test_loss_mean = All_test_loss / len(test_loader)
            Test_acc_mean = All_test_acc / len(Test_dataset)
            print(f"Test task loss: {Test_loss_mean:.4f}" + '-----'
                  + f"Test task acc:  {Test_acc_mean:.4f}")

            if Test_loss_mean <= curr_test_loss:
                curr_test_loss = Test_loss_mean
                torch.save(model.state_dict(),
                           f"{l1}_{Dataset}_{inner_step}_{inner_lr}_Way_{10}_Shot_{5}_{initial_margin}.pth")
                print("Model parameters saved.")

            changed = False
            for key in initial_state_dict:
                if not torch.equal(initial_state_dict[key], model.state_dict()[key]):
                    print(f"Parameter {key} has changed.")
                    changed = True

            if not changed:
                print("All parameters are unchanged.")
            else:
                print("Some parameters were modified!")

            print("===================================Test Task over===================================")




