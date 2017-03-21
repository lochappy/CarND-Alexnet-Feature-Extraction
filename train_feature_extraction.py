import pickle, os, shutil, time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np
from sklearn.utils import shuffle

# TODO: Load traffic signs data.
with open('train.p','rb') as f:
    trainData = pickle.load(f) 
    
n_classes = len(set(trainData['labels']))
# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(
    trainData['features'].astype(np.float32), trainData['labels'].astype(np.int64), test_size=0.2, random_state=42)

# TODO: Define placeholders and resize operation.
inputData = tf.placeholder(tf.float32,shape=[None,32,32,3],name='inputData')
inputLabel = tf.placeholder(tf.int64,shape=[None],name='inputData')

y_one_hot = tf.one_hot(inputLabel,n_classes,name='y_one_hot')

resized = tf.image.resize_images(inputData, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], n_classes)  # use this shape for the weight matrix
w = tf.Variable(tf.truncated_normal(shape), name='w_final')
b = tf.Variable(tf.zeros([n_classes]), name='b_final')
logits = tf.nn.xw_plus_b(fc7,w,b,name='logits')
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_one_hot), name='loss')
loss_on_trainset_summary = tf.summary.scalar('LossOnTrainset', loss)
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(probs,axis=1),tf.argmax(y_one_hot,axis=1)),tf.float32), name='accuracy')


def evaluate(X_data, y_data, sess, BATCH_SIZE):
    num_examples = len(X_data)
    total_accuracy = 0.
    total_loss = 0.
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(acc, feed_dict={ inputData: batch_x, inputLabel: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    total_accuracy /= num_examples
    sess.run(acc_on_valset.assign(total_accuracy))
    return total_accuracy

with tf.name_scope('EvaluationOnValSet'):
    acc_on_valset = tf.Variable(0,dtype=tf.float32, trainable=False)
acc_on_valset_summary = tf.summary.scalar('AccOnValset', acc_on_valset)



lr = 2e-3
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# TODO: Train and evaluate the feature extraction model.

saver = tf.train.Saver()
EPOCHS = 100
BATCH_SIZE = 32

sess = tf.Session()

if os.path.exists('ConvNetOnTrafficSignGraph'):
    shutil.rmtree('ConvNetOnTrafficSignGraph')
    
if os.path.exists('ConvNetOnTrafficSign'):
    shutil.rmtree('ConvNetOnTrafficSign')
os.mkdir('ConvNetOnTrafficSign')

init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(init)
num_examples = len(X_train)
train_writer = tf.summary.FileWriter('./ConvNetOnTrafficSignGraph/train', sess.graph)
test_writer = tf.summary.FileWriter('./ConvNetOnTrafficSignGraph/test')
print("Training...")

count = 0
max_validation_accuracy = 0.0
    
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    
maxIdx = 0
for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    t0 = time.time()
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        _, summary = sess.run([optimizer,loss_on_trainset_summary], feed_dict={ inputData: batch_x, inputLabel: batch_y})
        train_writer.add_summary(summary,global_step=count)
        count += 1
    train_writer.flush() 
    t0 = time.time() - t0
    
    validation_accuracy = evaluate(X_test, y_test, sess, BATCH_SIZE)
    test_writer.add_summary(sess.run(acc_on_valset_summary),global_step=i)
    test_writer.flush()
        
    if (validation_accuracy > max_validation_accuracy):
        saver.save(sess, save_path='./ConvNetOnTrafficSign/Model',global_step=i)
        max_validation_accuracy = validation_accuracy
        maxIdx = i
        
    print("EPOCH {}\tValidation Accuracy = {:.3f}\tTime={:.3f}\tMaxAcc = {:.3f} @ epoch {}".format(i, validation_accuracy, t0, max_validation_accuracy, maxIdx))
    
coord.request_stop()
coord.join(threads)
sess.close()
