from model_back import *
import sys
sys.path.append("~/Desktop/iort/utils/")
from logging import Logger
# Variables

BATCH_SIZE = 128
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
START_EPOCH = 0
NUM_EPOCHS = 200
VAL_FREQUENCY = 10
SAVE_FREQUENCY = 1000
TRAIN_PERCENT = 0.7

DATA_DIR = "data"
LOG_DIR = 'log_files/'
SAVE_DIR = './simple_model/'
RUN_NAME = 'simple_model'
MODEL_NAME = 'model_'
SAVE_PATH = SAVE_DIR + MODEL_NAME

def main():

    print('Loading TrainVal Data')
    """ TODO: Add train_val_location """
    tensor_logs = Logger(LOG_DIR)
    print('Logger object created')
    train_loader, val_loader = get_trainval_data(batch_size=BATCH_SIZE, train_percent=TRAIN_PERCENT)

    """ TODO: Add test data """
    # print('Loading Test Data')
    # test_data_loc = osp.join(DATA_DIR, "randomized_annotated_test_set_no_name_no_num.p")
    # test_dataset, test_loader = get_test_data(test_data_loc, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    model = Model(im_size=IM_SIZE)

    if(IS_CUDA):
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    print('Started training...')

    for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
        train(train_loader, model, criterion, optimizer, epoch, tensor_logs)
        if epoch % VAL_FREQUENCY == 0:
            validate(val_loader, model, epoch)

if __name__ == '__main__':
    main()
