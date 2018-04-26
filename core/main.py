from model_back import *
import sys
from preprocess import process_test, resize_img
import cv2
sys.path.append("../utils/")
from logger import Logger
# Variables

BATCH_SIZE = 16
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
START_EPOCH = 0
NUM_EPOCHS = 5
VAL_FREQUENCY = 10
SAVE_FREQUENCY = 1000
TRAIN_PERCENT = 0.7

DATA_DIR = "data"
TESTDATA_DIR = "test_data"
LOG_DIR = 'log_files/'
SAVE_DIR = './'
RUN_NAME = 'simple_model'
MODEL_NAME = 'latest_model'
SAVE_FREQ = 10
SAVE_PATH = SAVE_DIR + MODEL_NAME
SAVE_HEAT_PATH = './heat_maps/'

def main():

    print('Loading TrainVal Data')
    tensor_logs = Logger(LOG_DIR, name="abc")  
    train_loader, val_loader = get_trainval_data(batch_size=BATCH_SIZE, train_percent=TRAIN_PERCENT, data_dir='../data/', num_dir=1)

    print('Loading Test Data')
    test_dataset, img_shapes = get_test_data(data_dir='../test_data/', num_dir=2)

    # print("Test data shape", np.asarray(test_dataset[0][0][0]).shape)
    
    model = Model(im_size=IM_SIZE)

    if(IS_CUDA):
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    print('Started training...')

    for epoch in range(START_EPOCH, START_EPOCH + NUM_EPOCHS):
        train(train_loader, model, criterion, optimizer, epoch, tensor_logs)
        
        if epoch % VAL_FREQUENCY == 0 and epoch!=0:
            validate(val_loader, model, epoch)

        """ TODO: Implement save model """
        # if epoch% SAVE_FREQ == 0:
            # torch.save(model, SAVE_PATH)

    heat_maps = []

    print('Testing model...')
    for i, (test_data, img_shape) in enumerate(zip(test_dataset, img_shapes)):
        # inside one image
        predictions = test(test_data, model)

        location_data = [x[1] for x in test_data]
        heat_map = process_test(img_shape, location_data, predictions)
        cv2.imwrite(SAVE_HEAT_PATH+str(i)+'.png',heat_map)
        
        """ book-keeping """
        heat_maps.append(heat_map)

    ## show heat_map
    # cv2.imwrite('temp.png',heat_maps[1])
        

if __name__ == '__main__':
    main()













