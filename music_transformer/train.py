from model import MusicTransformer
import custom
from custom.metrics import *
from custom.criterion import SmoothCrossEntropyLoss, CustomSchedule
from custom.config import config
from data import Data

import utils
import datetime
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import traceback


# set config
parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')
    
print('Summary - Device Info :')
print('CUDA is available: {}'.format(torch.cuda.is_available()))
print('Number of GPUs: {}'.format(torch.cuda.device_count()))

for i in range (0,torch.cuda.device_count()):
    print('GPU #{}: {}'.format(i,torch.cuda.get_device_name(i)))
    print('Memory Usage:')
    print('Allocated: ', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    
# load data
dataset = Data(config.pickle_dir)
print(dataset)


# load model
learning_rate = config.l_r

# define model
mt = MusicTransformer(
            embedding_dim=config.embedding_dim,
            vocab_size=config.vocab_size,
            num_layer=config.num_layers,
            max_seq=config.max_seq,
            dropout=config.dropout,
            debug=config.debug, loader_path=config.load_path
)
mt.to(config.device)
opt = optim.Adam(mt.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
scheduler = CustomSchedule(config.embedding_dim, optimizer=opt)

# multi-GPU set
if torch.cuda.device_count() > 1:
    single_mt = mt
    mt = torch.nn.DataParallel(mt, output_device=torch.cuda.device_count()-1)
else:
    single_mt = mt

# init metric set
metric_set = MetricsSet({
    'accuracy': CategoricalAccuracy(),
    'loss': SmoothCrossEntropyLoss(config.label_smooth, config.vocab_size, config.pad_token),
    'bucket':  LogitsBucketting(config.vocab_size),
    'nll': NegativeLogLoss(),
    'pitch_entropy': PitchCrossEntropy(),
})
print('PyTorch Model Summary:\n {}'.format(mt))

# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/'+config.experiment+'/'+current_time+'/train'
eval_log_dir = 'logs/'+config.experiment+'/'+current_time+'/eval'

train_summary_writer = SummaryWriter(train_log_dir)
eval_summary_writer = SummaryWriter(eval_log_dir)

# Train Start
print(">> Train start...")
idx = 0
error_num = 0
success = 0
for e in range(config.epochs):
    print("Epoch: {}".format(e))
    for b in range(len(dataset.files) // config.batch_size):
        scheduler.optimizer.zero_grad()
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(config.batch_size, config.max_seq)
            batch_x = torch.from_numpy(batch_x).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
            batch_y = torch.from_numpy(batch_y).contiguous().to(config.device, non_blocking=True, dtype=torch.int)
        except IndexError:
            if error_num % 100 == 0:
                traceback.print_exc()
            error_num += 1
            continue
        print("b is : {}".format(b))
        start_time = time.time()
        mt.train()
        sample = mt.forward(batch_x)
        sample = list(sample)
        #print(len(sample))
        #print(sample[0])
       
        #for i in sample:
            #print(i)
            #print()
            #print('-'*60)
            #print()
        #print(batch_y)
        metrics = metric_set(sample[0], batch_y)
        loss = metrics['loss']  
        loss.backward()
        scheduler.step()
        end_time = time.time()
        success += 1
        if config.debug:
            print("[Loss]: {}".format(loss))

        train_summary_writer.add_scalar('loss', metrics['loss'], global_step=idx)
        train_summary_writer.add_scalar('pitch_entropy', metrics['pitch_entropy'], global_step=idx)
        train_summary_writer.add_scalar('nll', metrics['nll'], global_step=idx)
        train_summary_writer.add_scalar('accuracy', metrics['accuracy'], global_step=idx)
        train_summary_writer.add_scalar('learning_rate', scheduler.rate(), global_step=idx)
        train_summary_writer.add_scalar('iter_p_sec', end_time-start_time, global_step=idx)

        # result_metrics = metric_set(sample, batch_y)
        if b % 100 == 0:
            single_mt.eval()
            eval_x, eval_y = dataset.slide_seq2seq_batch(2, config.max_seq, 'eval')
            eval_x = torch.from_numpy(eval_x).contiguous().to(config.device, dtype=torch.int)
            eval_y = torch.from_numpy(eval_y).contiguous().to(config.device, dtype=torch.int)

            eval_preiction, weights = single_mt.forward(eval_x)

            eval_metrics = metric_set(eval_preiction, eval_y)
            torch.save(single_mt.state_dict(), args.model_dir+'/train-{}.pth'.format(e))
            if b == 0:
                train_summary_writer.add_histogram("target_analysis", batch_y, global_step=e)
                train_summary_writer.add_histogram("source_analysis", batch_x, global_step=e)
                for i, weight in enumerate(weights):
                    attn_log_name = "attn/layer-{}".format(i)
                    utils.attention_image_summary(
                        attn_log_name, weight, step=idx, writer=eval_summary_writer)
            
           
            eval_summary_writer.add_scalar('loss', eval_metrics['loss'], global_step=idx)
            eval_summary_writer.add_scalar('pitch_entropy', eval_metrics['pitch_entropy'], global_step=idx)
            eval_summary_writer.add_scalar('nll', eval_metrics['nll'], global_step=idx)
            eval_summary_writer.add_scalar('accuracy', eval_metrics['accuracy'], global_step=idx)
            eval_summary_writer.add_histogram("logits_bucket", eval_metrics['bucket'], global_step=idx)

            print('\n====================================================')
            print('Epoch/Batch: {}/{}'.format(e, b))
            print('Train >>>> Loss: {:6.6}, Pitch Entropy: {:6.6}, Negative Log: {:6.6}, Accuracy: {}'.format(metrics['loss'], metrics['pitch_entropy'], metrics['nll'], metrics['accuracy']))
            print('Eval Train >>>> Loss: {:6.6}, Pitch Entropy: {:6.6}, Negative Log: {:6.6}, Accuracy: {}'.format(eval_metrics['loss'], eval_metrics['pitch_entropy'], eval_metrics['nll'], eval_metrics['accuracy']))
        torch.cuda.empty_cache()
        idx += 1

        # switch output device to: gpu-1 ~ gpu-n
        sw_start = time.time()
        mt.output_device = idx % (torch.cuda.device_count() -1) + 1
        sw_end = time.time()
        if config.debug:
            print('output switch time: {}'.format(sw_end - sw_start) )

print('Errors: {}\tSuccesses: {}'.format(error_num,success))
torch.save(single_mt.state_dict(), args.model_dir+'/final.pth'.format(idx))
eval_summary_writer.close()
train_summary_writer.close()


