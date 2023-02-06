import torch


@profile
def main():
    aa = torch.randn(1000, 1000000)
    bb = torch.randn(1, 1000000)

    cc = aa.unsqueeze(1) * bb.unsqueeze(0)

if __name__ == '__main__':
    main()



