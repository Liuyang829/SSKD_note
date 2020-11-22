import os
import os.path as osp
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from utils import AverageMeter, accuracy
from wrapper import wrapper
from cifar import CIFAR100

from models import model_dict

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='train SSKD student network.')
parser.add_argument('--epoch', type=int, default=240)
parser.add_argument('--t-epoch', type=int, default=60)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--t-lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[150,180,210])
parser.add_argument('--t-milestones', type=int, nargs='+', default=[30,45])

parser.add_argument('--save-interval', type=int, default=40)
parser.add_argument('--ce-weight', type=float, default=0.1) # cross-entropy
parser.add_argument('--kd-weight', type=float, default=0.9) # knowledge distillation
parser.add_argument('--tf-weight', type=float, default=2.7) # transformation
parser.add_argument('--ss-weight', type=float, default=10.0) # self-supervision

parser.add_argument('--kd-T', type=float, default=4.0) # temperature in KD
parser.add_argument('--tf-T', type=float, default=4.0) # temperature in LT
parser.add_argument('--ss-T', type=float, default=0.5) # temperature in SS

parser.add_argument('--ratio-tf', type=float, default=1.0) # keep how many wrong predictions of LT
parser.add_argument('--ratio-ss', type=float, default=0.75) # keep how many wrong predictions of SS
parser.add_argument('--s-arch', type=str) # student architecture
parser.add_argument('--t-path', type=str) # teacher checkpoint path

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu-id', type=int, default=0)

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


t_name = osp.abspath(args.t_path).split('/')[-1]
t_arch = '_'.join(t_name.split('_')[1:-1])
exp_name = 'sskd_student_{}_weight{}+{}+{}+{}_T{}+{}+{}_ratio{}+{}_seed{}_{}'.format(\
            args.s_arch, \
            args.ce_weight, args.kd_weight, args.tf_weight, args.ss_weight, \
            args.kd_T, args.tf_T, args.ss_T, \
            args.ratio_tf, args.ratio_ss, \
            args.seed, t_name)
exp_path = './experiments/{}'.format(exp_name)
os.makedirs(exp_path, exist_ok=True)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
])

trainset = CIFAR100('~/data', train=True, transform=transform_train)
valset = CIFAR100('~/data', train=False, transform=transform_test)

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

ckpt_path = osp.join(args.t_path, 'ckpt/best.pth')
t_model = model_dict[t_arch](num_classes=100).cuda()
state_dict = torch.load(ckpt_path)['state_dict']
t_model.load_state_dict(state_dict)
# 这个wrapper
t_model = wrapper(module=t_model).cuda()

t_optimizer = optim.SGD([{'params':t_model.backbone.parameters(), 'lr':0.0},
                        {'params':t_model.proj_head.parameters(), 'lr':args.t_lr}],
                        momentum=args.momentum, weight_decay=args.weight_decay)
t_model.eval()
t_scheduler = MultiStepLR(t_optimizer, milestones=args.t_milestones, gamma=args.gamma)

logger = SummaryWriter(osp.join(exp_path, 'events'))

acc_record = AverageMeter()
loss_record = AverageMeter()
start = time.time()
for x, target in val_loader:

    # 把中间一个维度去掉,为什么会多一个维度还得看data
    x = x[:,0,:,:,:].cuda()
    target = target.cuda()
    with torch.no_grad():
        output, _, feat = t_model(x)
        loss = F.cross_entropy(output, target)

    batch_acc = accuracy(output, target, topk=(1,))[0]
    acc_record.update(batch_acc.item(), x.size(0))
    loss_record.update(loss.item(), x.size(0))

run_time = time.time() - start
info = 'teacher cls_acc:{:.2f}\n'.format(acc_record.avg)
print(info)

# train ssp_head
for epoch in range(args.t_epoch):

    t_model.eval()
    loss_record = AverageMeter()
    acc_record = AverageMeter()

    start = time.time()
    for x, _ in train_loader:

        t_optimizer.zero_grad()

        x = x.cuda()
        # print("x shape1:", x.shape) ([64, 4, 3, 32, 32])
        c,h,w = x.size()[-3:]
        # 这边就是把x变成正常?
        x = x.view(-1, c, h, w)
        # print("x shape2:", x.shape) ([256, 3, 32, 32])
        # bb_grad=False代表最后一层输出的feature不受梯度更新
        # feat是最后一层出来的feature向量，rep是投影头的输出结果，自监督模块
        _, rep, feat = t_model(x, bb_grad=False)
        # print(k.shape)

        # 正常batch
        batch = int(x.size(0) / 4)
        # 对应normal与augment
        # 因为每张图变成了4张，只有第一张是原图，后三张都变成了变换过的，所以余数=0代表原图，其他都是aug
        nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
        aug_index = (torch.arange(4*batch) % 4 != 0).cuda()
        # print(nor_index, aug_index)

        nor_rep = rep[nor_index]
        aug_rep = rep[aug_index]
        # print(aug_rep.shape) [192,128] 数量不一样 数量是正常的3倍
        # print(nor_rep.shape) [64,128] 64个128维
        # nor_rep通过unsqueece增加第二个维度，变[64,128,1]
        # 再通过expand变成[64,128,192] 用相同的数值元素扩展至3倍
        # 通过transpose转置了下变成[192,128,64]
        nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
        # print(nor_rep.shape) [192,128,64]
        aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
        # print(aug_rep.shape) 本来就是192个，batch为64[192,128,64]
        simi = F.cosine_similarity(aug_rep, nor_rep, dim=1) #算Aij
        # target变成[0,0,0,1,1,1,2,2,2……,62,62,62,63,63,63]
        target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
        loss = F.cross_entropy(simi, target)

        loss.backward()
        t_optimizer.step()

        batch_acc = accuracy(simi, target, topk=(1,))[0]
        loss_record.update(loss.item(), 3*batch)
        acc_record.update(batch_acc.item(), 3*batch)

    logger.add_scalar('train/teacher_ssp_loss', loss_record.avg, epoch+1)
    logger.add_scalar('train/teacher_ssp_acc', acc_record.avg, epoch+1)

    run_time = time.time() - start
    info = 'teacher_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\t'.format(
        epoch+1, args.t_epoch, run_time, loss_record.avg, acc_record.avg)
    print(info)

    t_model.eval()
    acc_record = AverageMeter()
    loss_record = AverageMeter()
    start = time.time()
    for x, _ in val_loader:

        x = x.cuda()
        c,h,w = x.size()[-3:]
        x = x.view(-1, c, h, w)
        # 和训练部分基本一样
        with torch.no_grad():
            _, rep, feat = t_model(x)
        batch = int(x.size(0) / 4)
        nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
        aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

        nor_rep = rep[nor_index]
        aug_rep = rep[aug_index]
        nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
        aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
        simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
        target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
        loss = F.cross_entropy(simi, target)

        batch_acc = accuracy(simi, target, topk=(1,))[0]
        acc_record.update(batch_acc.item(),3*batch)
        loss_record.update(loss.item(), 3*batch)

    run_time = time.time() - start
    logger.add_scalar('val/teacher_ssp_loss', loss_record.avg, epoch+1)
    logger.add_scalar('val/teacher_ssp_acc', acc_record.avg, epoch+1)

    info = 'ssp_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\n'.format(
            epoch+1, args.t_epoch, run_time, loss_record.avg, acc_record.avg)
    print(info)

    t_scheduler.step()


name = osp.join(exp_path, 'ckpt/teacher.pth')
os.makedirs(osp.dirname(name), exist_ok=True)
torch.save(t_model.state_dict(), name)


s_model = model_dict[args.s_arch](num_classes=100)
s_model = wrapper(module=s_model).cuda()
optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

best_acc = 0
for epoch in range(args.epoch):

    # train
    s_model.train()
    loss1_record = AverageMeter()
    loss2_record = AverageMeter()
    loss3_record = AverageMeter()
    loss4_record = AverageMeter()
    cls_acc_record = AverageMeter()
    ssp_acc_record = AverageMeter()
    
    start = time.time()
    for x, target in train_loader:

        optimizer.zero_grad()

        c,h,w = x.size()[-3:]
        x = x.view(-1,c,h,w).cuda() #([256, 3, 32, 32])
        target = target.cuda()

        batch = int(x.size(0) / 4)
        nor_index = (torch.arange(4*batch) % 4 == 0).cuda() #[256]
        aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

        # True代表最后一层受梯度更新
        output, s_feat, _ = s_model(x, bb_grad=True)
        # output.shape [256,100]
        # 为什么对应两个不同的温度 normal对应kd aug对应transf·········
        log_nor_output = F.log_softmax(output[nor_index] / args.kd_T, dim=1)  #[64,100]
        log_aug_output = F.log_softmax(output[aug_index] / args.tf_T, dim=1)  #[192,100]
        with torch.no_grad():
            # 对于教师的结果 也带上温度 knowledge对应的是backbone预测结果 shape为 [256,100]
            knowledge, t_feat, _ = t_model(x)
            nor_knowledge = F.softmax(knowledge[nor_index] / args.kd_T, dim=1)  #[64,100]
            aug_knowledge = F.softmax(knowledge[aug_index] / args.tf_T, dim=1)  #[192,100]
        # error level ranking
        # 根据target生成对应的3倍扩展后列表[192] 对应aug_data
        aug_target = target.unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
        # 将预测结果排序 生成相同维度的rank序列 [192,100] 将降序后的下标进行从大到小排序 排在第一个的就是概率值最高的那个下标
        # rank对应的就是每个样本预测结果的一个排名
        rank = torch.argsort(aug_knowledge, dim=1, descending=True)
        # rank是[192,100],aug_target是[192,1]
        # torch.eq是继续生成了一个[192,100]的矩阵，相等的地方标1，不相等为0 看和groundtruth是否预测正确，或者是概率值排第几
        # torch.argmax返回了每个为1的点的下标，生成一个[192]，代表每个样本第几个概率值预测正确 0就代表预测正确 10就代表排第10的那个才正确
        rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
        # 将所有预测结果的顺序继续从大到小排，显示下标，为0的预测正确的都排在前面，从前往后分别就是 正确的-错误率越来越大的 生成index
        index = torch.argsort(rank)
        # nonzero()返回rank中非零点的坐标，预测正确的都与第0个直接对应所以都是0，非0代表没有预测正确的
        tmp = torch.nonzero(rank, as_tuple=True)[0]
        wrong_num = tmp.numel() #numel() 返回元素个数
        correct_num = 3*batch - wrong_num
        # 只保留一部分错误预测结果 但是ratio_tf为1.0
        wrong_keep = int(wrong_num * args.ratio_tf)
        # 保留正确和前一部分错误的 但是这边是全部保留了反正
        index = index[:correct_num+wrong_keep]
        # 将所有保留下来的进行一个排序
        distill_index_tf = torch.sort(index)[0]

        # s_feat是投影头 两个Linear生成的结果
        s_nor_feat = s_feat[nor_index]
        s_aug_feat = s_feat[aug_index]
        s_nor_feat = s_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
        s_aug_feat = s_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
        s_simi = F.cosine_similarity(s_aug_feat, s_nor_feat, dim=1)

        t_nor_feat = t_feat[nor_index]
        t_aug_feat = t_feat[aug_index]
        t_nor_feat = t_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
        t_aug_feat = t_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
        t_simi = F.cosine_similarity(t_aug_feat, t_nor_feat, dim=1)

        # 教师的不变 下面的ss模块的teacher处理和上面类似
        t_simi = t_simi.detach()
        aug_target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
        # 将教师中的相似度排序 保留下标
        rank = torch.argsort(t_simi, dim=1, descending=True)
        rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)  # groundtruth label's rank
        index = torch.argsort(rank)
        tmp = torch.nonzero(rank, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = 3*batch - wrong_num
        wrong_keep = int(wrong_num * args.ratio_ss)
        index = index[:correct_num+wrong_keep]
        # 选取了一部分我们要的 teacher ss的结果保留
        distill_index_ss = torch.sort(index)[0]

        log_simi = F.log_softmax(s_simi / args.ss_T, dim=1)
        simi_knowledge = F.softmax(t_simi / args.ss_T, dim=1)

        # 论文中的四个loss 第一个普通loss 预测结果与ground truth
        loss1 = F.cross_entropy(output[nor_index], target)
        # output是student knowledge是teacher出来的
        # log_nor_output = F.log_softmax(output[nor_index] / args.kd_T, dim=1)  # [64,100]
        # nor_knowledge = F.softmax(knowledge[nor_index] / args.kd_T, dim=1)  # [64,100]
        # 知识蒸馏的loss log_nor_output是学生对于正常图片结果除以温度后计算log_softmax
        # nor_knowledge是teacher对于正常图片除以蒸馏温度后计算softmax
        loss2 = F.kl_div(log_nor_output, nor_knowledge, reduction='batchmean') * args.kd_T * args.kd_T
        # log_aug_output = F.log_softmax(output[aug_index] / args.tf_T, dim=1) # [192,100]
        # aug_knowledge = F.softmax(knowledge[aug_index] / args.tf_T, dim=1)  # [192,100]
        # transform的loss log_nor_output是学生对于变换后图片结果除以变换温度后计算log_softmax
        # aug_knowledge是teacher对于变换后图片除以变换温度后计算softmax [distill_index_tf]目前是包括所有的
        loss3 = F.kl_div(log_aug_output[distill_index_tf], aug_knowledge[distill_index_tf], \
                        reduction='batchmean') * args.tf_T * args.tf_T
        # 自监督loss 学生的相似度算log 与 teacher的相似度 这里只保留了前百分之多少的结果
        loss4 = F.kl_div(log_simi[distill_index_ss], simi_knowledge[distill_index_ss], \
                        reduction='batchmean') * args.ss_T * args.ss_T

        loss = args.ce_weight * loss1 + args.kd_weight * loss2 + args.tf_weight * loss3 + args.ss_weight * loss4

        loss.backward()
        optimizer.step()

        cls_batch_acc = accuracy(output[nor_index], target, topk=(1,))[0]
        ssp_batch_acc = accuracy(s_simi, aug_target, topk=(1,))[0]
        loss1_record.update(loss1.item(), batch)
        loss2_record.update(loss2.item(), batch)
        loss3_record.update(loss3.item(), len(distill_index_tf))
        loss4_record.update(loss4.item(), len(distill_index_ss))
        cls_acc_record.update(cls_batch_acc.item(), batch)
        ssp_acc_record.update(ssp_batch_acc.item(), 3*batch)

    logger.add_scalar('train/ce_loss', loss1_record.avg, epoch+1)
    logger.add_scalar('train/kd_loss', loss2_record.avg, epoch+1)
    logger.add_scalar('train/tf_loss', loss3_record.avg, epoch+1)
    logger.add_scalar('train/ss_loss', loss4_record.avg, epoch+1)
    logger.add_scalar('train/cls_acc', cls_acc_record.avg, epoch+1)
    logger.add_scalar('train/ss_acc', ssp_acc_record.avg, epoch+1)

    run_time = time.time() - start
    info = 'student_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t ce_loss:{:.3f}\t kd_loss:{:.3f}\t cls_acc:{:.2f}'.format(
        epoch+1, args.epoch, run_time, loss1_record.avg, loss2_record.avg, cls_acc_record.avg)
    print(info)

    # cls val
    s_model.eval()
    acc_record = AverageMeter()
    loss_record = AverageMeter()
    start = time.time()
    for x, target in val_loader:

        x = x[:,0,:,:,:].cuda()
        target = target.cuda()
        with torch.no_grad():
            output, _, feat = s_model(x)
            loss = F.cross_entropy(output, target)

        batch_acc = accuracy(output, target, topk=(1,))[0]
        acc_record.update(batch_acc.item(), x.size(0))
        loss_record.update(loss.item(), x.size(0))

    run_time = time.time() - start
    logger.add_scalar('val/ce_loss', loss_record.avg, epoch+1)
    logger.add_scalar('val/cls_acc', acc_record.avg, epoch+1)

    info = 'student_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t cls_acc:{:.2f}\n'.format(
            epoch+1, args.epoch, run_time, acc_record.avg)
    print(info)

    if acc_record.avg > best_acc:
        best_acc = acc_record.avg
        state_dict = dict(epoch=epoch+1, state_dict=s_model.state_dict(), best_acc=best_acc)
        name = osp.join(exp_path, 'ckpt/student_best.pth')
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(state_dict, name)
    
    scheduler.step()

