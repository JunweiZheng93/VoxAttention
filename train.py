import tensorflow as tf
from utils import dataloader
from datetime import datetime
from hparam import hparam
import model
import os


PROJ_ROOT = os.path.abspath(__file__)[:-8]
RESULT_PATH = os.path.join(PROJ_ROOT, 'results', datetime.now().strftime("%Y%m%d%H%M%S"))


def train_model(category='chair',
                batch_size=32,
                split_ratio=0.8,
                max_num_parts=4,
                bce_weight=0.6,
                bce_weight_shape=0.6,
                optimizer='adam',
                lr=0.001,
                decay_rate=0.8,
                decay_step_size=50,
                training_process=1,
                epochs=500,
                model_path=None,
                shuffle=True,
                use_attention=False,
                keep_channel=False,
                use_ac_loss=False,
                which_layer='0',
                num_blocks=6,
                num_heads=8,
                d_model=256,
                which_gpu=0):

    # disable warning and info message, only enable error message
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    _configure_gpu(which_gpu)
    _save_necessary_scripts(RESULT_PATH)

    # get dataset
    training_set, test_set = dataloader.get_dataset(category=category, batch_size=batch_size, split_ratio=split_ratio, max_num_parts=max_num_parts)
    # create model
    my_model = model.Model(max_num_parts, bce_weight, bce_weight_shape, training_process, use_attention, keep_channel, use_ac_loss, which_layer, num_blocks, num_heads, d_model)

    if training_process == 1 or training_process == '1':
        _execute_training_process(my_model, training_set, test_set, epochs, shuffle, 1, use_attention, use_ac_loss, optimizer, lr, decay_rate, decay_step_size, RESULT_PATH)

    elif training_process == 2 or training_process == '2':
        warm_up_data = training_set.__iter__().__next__()[0]
        my_model(warm_up_data)
        my_model.load_weights(model_path, by_name=True)
        _execute_training_process(my_model, training_set, test_set, epochs, shuffle, 2, use_attention, use_ac_loss, optimizer, lr, decay_rate, decay_step_size, RESULT_PATH)

    elif training_process == 3 or training_process == '3':
        warm_up_data = training_set.__iter__().__next__()[0]
        my_model(warm_up_data)
        my_model.load_weights(model_path, by_name=True)
        _execute_training_process(my_model, training_set, test_set, epochs, shuffle, 3, use_attention, use_ac_loss, optimizer, lr, decay_rate, decay_step_size, RESULT_PATH)

    else:
        raise ValueError('training_process should be one of 1, 2 and 3')


def _configure_gpu(which_gpu):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[which_gpu], "GPU")


def _save_necessary_scripts(result_saved_path):

    # make directory and save __init__.py
    if not os.path.exists(result_saved_path):
        os.makedirs(result_saved_path)
        result_init_saved_path = os.path.join(os.path.dirname(result_saved_path), '__init__.py')
        os.system(f'touch {result_init_saved_path}')
        experiment_init_saved_path = os.path.join(result_saved_path, '__init__.py')
        os.system(f'touch {experiment_init_saved_path}')

    # save hparam.py
    hparam_saved_path = os.path.join(result_saved_path, 'hparam.py')
    hparam_source_path = os.path.join(PROJ_ROOT, 'hparam.py')
    os.system(f'cp {hparam_source_path} {hparam_saved_path}')

    # save model.py
    model_saved_path = os.path.join(result_saved_path, 'model.py')
    model_source_path = os.path.join(PROJ_ROOT, 'model.py')
    os.system(f'cp {model_source_path} {model_saved_path}')

    # save train.py
    train_saved_path = os.path.join(result_saved_path, 'train.py')
    train_source_path = os.path.join(PROJ_ROOT, 'train.py')
    os.system(f'cp {train_source_path} {train_saved_path}')


def _get_optimizer(opt, lr):
    if opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr)
    elif opt == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr)
    else:
        raise ValueError(f'{opt} is not valid!')
    return optimizer


def _get_lr_scheduler(decay_rate, decay_step_size):
    def lr_scheduler(epoch, lr):
        if (epoch + 1) % decay_step_size == 0:
            lr = lr * (1 - decay_rate)
        return lr
    return lr_scheduler


class EvaluationCallback(tf.keras.callbacks.Callback):

    def __init__(self, test_set, summary_writer):
        self.test_set = test_set
        self.summary_writer = summary_writer

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            self.model.evaluate(self.test_set)
            with self.summary_writer.as_default():
                if self.model.training_process == 1 or self.model.training_process == '1':
                    for i in range(self.model.num_parts):
                        tf.summary.scalar(f'eval_Part{i+1}_mIoU', self.model.part_mIoU_tracker_list[i].result(), step=epoch)
                    tf.summary.scalar('eval_Part_mIoU', self.model.all_part_mIoU_tracker.result(), step=epoch)
                elif self.model.training_process == 2 or self.model.training_process == '2':
                    tf.summary.scalar('eval_Transformation_MSE', self.model.transformation_mse_tracker.result(), step=epoch)
                    tf.summary.scalar('eval_Shape_mIoU', self.model.shape_mIoU_tracker.result(), step=epoch)
                else:
                    for i in range(self.model.num_parts):
                        tf.summary.scalar(f'eval_Part{i+1}_mIoU', self.model.part_mIoU_tracker_list[i].result(), step=epoch)
                    tf.summary.scalar('eval_Part_mIoU', self.model.all_part_mIoU_tracker.result(), step=epoch)
                    tf.summary.scalar('eval_Transformation_MSE', self.model.transformation_mse_tracker.result(), step=epoch)
                    tf.summary.scalar('eval_Shape_mIoU', self.model.shape_mIoU_tracker.result(), step=epoch)


def _execute_training_process(my_model,
                              training_set,
                              test_set,
                              epochs,
                              shuffle,
                              process,
                              use_attention,
                              use_ac_loss,
                              optimizer,
                              lr,
                              decay_rate,
                              decay_step_size,
                              result_saved_path):
    print(f'Start training process {process}, please wait...')
    process_saved_path = os.path.join(result_saved_path, f'process_{process}')
    if not os.path.exists(process_saved_path):
        os.mkdir(process_saved_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(process_saved_path, 'checkpoint.h5'),
                                                             monitor='Transformation_Loss' if (process == 2 and use_attention and not use_ac_loss) or
                                                                                              (process == 2 and not use_attention) else 'Total_Loss',
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode='min')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(process_saved_path, 'logs'), profile_batch=0)
    summary_writer = tf.summary.create_file_writer(os.path.join(process_saved_path, 'logs', 'validation'))
    lr_scheduler = _get_lr_scheduler(decay_rate, decay_step_size)
    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    evaluation_callback = EvaluationCallback(test_set, summary_writer)
    callbacks = [checkpoint_callback, tensorboard_callback, lr_scheduler_callback, evaluation_callback]
    opt = _get_optimizer(optimizer, lr)
    my_model.compile(optimizer=opt, run_eagerly=True)
    my_model.fit(training_set, epochs=epochs, callbacks=callbacks, shuffle=shuffle)


if __name__ == '__main__':

    train_model(category=hparam['category'],
                batch_size=hparam['batch_size'],
                split_ratio=hparam['split_ratio'],
                max_num_parts=hparam['max_num_parts'],
                bce_weight=hparam['bce_weight'],
                bce_weight_shape=hparam['bce_weight_shape'],
                optimizer=hparam['optimizer'],
                lr=hparam['lr'],
                decay_rate=hparam['decay_rate'],
                decay_step_size=hparam['decay_step_size'],
                training_process=hparam['training_process'],
                epochs=hparam['epochs'],
                model_path=hparam['model_path'],
                shuffle=hparam['shuffle'],
                use_attention=hparam['use_attention'],
                keep_channel=hparam['keep_channel'],
                use_ac_loss=hparam['use_ac_loss'],
                which_layer=hparam['which_layer'],
                num_blocks=hparam['num_blocks'],
                num_heads=hparam['num_heads'],
                d_model=hparam['d_model'],
                which_gpu=hparam['which_gpu'])
