import model
import util

def main():
    train_ds, val_ds, train_ds = util.parse_dataset()

    # Outline: 
    # 2 different fully-connected architectures
    #   - At least 1 hidden layer
    #   - Hidden layers use ReLU
    #   - Output layers uses softmax
    # Adam as the optimizer
    # 2 different hyperparameters: (i.e., learning rates, batch size)
    # 2 different reglularizations (i.e., dropout, L2 regularization)

    # Next todo: make model class take in these different parameters and make a model the way we like?

if __name__ == '__main__':
    main()