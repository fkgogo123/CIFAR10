import torch
from torch import nn
from torch.nn import functional as F


class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """

    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv_unit = nn.Sequential(
            # x:[b, 3, 32, 32] => [b, 6, #, #]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            #
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #

        )
        # faltten
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        tmp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        print("conv out:", out.shape)

        # use Cross Entropy Loss
        # 一般对于分类问题使用Cross Entropy
        # 对于逼近或者regression的问题,使用MSE
        # self.criteon = nn.CrossEntropyLoss()

    def forward(self, x):
        """

        :param x: [b, 3, 32, 32]
        :return:
        """
        batchsz = x.size(0)  # x.shape[0]
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)
        # [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(batchsz, 16 * 5 * 5)  # -1
        # [b, 16*5*5] => [b, 10]
        logits = self.fc_unit(x)  # softmax之前的变量一般称为logits
        # softmax会有个数据不稳定,所以把它包含在了cross_entropy之中

        # [b, 10]
        # 因为CER中包含了softmax,所以下面这一步不需要写
        # pred = F.softmax(logits, dim=1)
        # loss在类外面做,就不用把y引进来,同时将上面的self.criteon注释
        # loss = self.criteon(logits, y)

        return logits


def main():
    net = Lenet5()

    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print("lenet out:", out.shape)


if __name__ == '__main__':
    main()
