# coding=gbk
import numpy as np
import struct

# 文件路径
train_images_idx3_ubyte_file = './q/train-images.idx3-ubyte'
train_labels_idx1_ubyte_file = './q/train-labels.idx1-ubyte'

test_images_idx3_ubyte_file = './q/t10k-images.idx3-ubyte'
test_labels_idx1_ubyte_file = './q/t10k-labels.idx1-ubyte'


# 读取文件
def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>IIII'  # 用二进制大端读取方法
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print("magic:%d, count: %d, size: %d*%d" % (magic_number, num_images, num_rows, num_cols))
    # 图像尺寸
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print("done %d" % (i + 1) + "pictures")
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print("magic:%d, num_images: %d zhang" % (magic_number, num_images))

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print("done %d" % (i + 1) + "zhang")
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


# 载入数据
def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


# 标准正则化 初始化参数
def narmalize_data(ima):
    a_max = np.max(ima)
    a_min = np.min(ima)
    for j in range(ima.shape[0]):
        ima[j] = (ima[j] - a_min) / (a_max - a_min)
    return ima


def initialize_with_zeros(n_x, n_h, n_y):
    np.random.seed(2)
    # W1=np.random.randn(n_h,n_x)
    W1 = np.random.randn(n_h, n_x) * 0.00000001
    # W1 = np.random.uniform(-np.sqrt(6) / np.sqrt(n_x + n_h), np.sqrt(6) / np.sqrt(n_h + n_x), size=(n_h, n_x))
    # W1=np.reshape(32,784)
    b1 = np.zeros((n_h, 1))
    # W2=np.random.randn(n_y,n_h)*0.00000001  # W2=np.random.randn(n_y,n_h)
    W2 = np.random.uniform(-np.sqrt(6) / np.sqrt(n_y + n_h), np.sqrt(6) / np.sqrt(n_y + n_h), size=(n_y, n_h))
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# 前向传播
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # print W1,X,b1
    Z1 = np.dot(W1, X) + b1
    # A1=sigmoid(Z1)
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    # assert(A2.shape == (1, X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache


# 损失函数的计算
def costloss(A2, Y, parameters):
    # m=Y.shape[0]
    t = 0.00000000001
    logprobs = np.multiply(np.log(A2 + t), Y) + np.multiply(np.log(1 - A2 + t), (1 - Y))
    # print("jixiaozhi: ",10*np.exp(-10))
    # logprobs = np.multiply(A2-Y,A2-Y)
    cost = np.sum(logprobs, axis=0, keepdims=True) / A2.shape[0]
    # cost=np.squeeze(cost)
    # assert(isinstance(cost, float))
    # cost=cost.astype(float)
    # cost=Variable(cost)
    return cost


# 反向传播
def back_propagation(parameters, cache, X, Y):
    # m=X.shape[0]
    # print('m',m)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]

    dZ2 = A2 - Y
    # print("dz2: ",dZ2)
    # print("A1: ",A1.T)
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    # dZ1=np.dot(W2.T,dZ2)*sigmoid(Z1)*(1-sigmoid(Z1))
    dW1 = np.dot(dZ1, X.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    # print("Dw2:",dW2)
    # print("Db2:",db2)
    return grads


# 更新参数
def update_para(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # 权重
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    print("learning_rate:", learning_rate)
    # 调节参数
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


# 定义sigmoid激活函数，softmax等参数。image2vector的作用是将输入从28*28的图片变成一个列向量
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


# 将输入从28*28的图片变成一个列向量
def image2vector(image):
    v = np.reshape(image, [784, 1])
    return v


# 返回最大数的索引
def softmax(x):
    v = np.argmax(x)
    return v


# 主函数
if __name__ == '__main__':
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    ii = 0
    n_x = 28 * 28  # 输入层
    n_h = 32  # 隐藏层
    n_y = 10  # 输出层
    # 初始化节点
    parameters = initialize_with_zeros(n_x, n_h, n_y)
    # 训练模型
    for i in range(50000):
        img_train = train_images[i]
        label_train1 = train_labels[i]
        label_train = np.zeros((10, 1))
        ttt = 0.001  # 学习率
        if i > 1000:
            ttt = ttt * 0.999
        label_train[int(train_labels[i])] = 1
        # print("train_label is: ", label_train)
        # print train_labels[i
        imgvector1 = image2vector(img_train)
        # print("imgvector1: before transform: ",imgvector1)
        imgvector = narmalize_data(imgvector1)
        # print("after transform: ",imgvector)

        # imgvector=image2vector(train_images)
        A2, cache = forward_propagation(imgvector, parameters)
        # print("A2:",A2)
        pre_label = softmax(A2)
        # print (pre_label, label_train1)
        # if pre_label==int(label_train1):
        # ii=ii+1
        # print("real value: ",label_train1)
        # print("pre_label: ",pre_label)
        costl = costloss(A2, label_train, parameters)
        grads = back_propagation(parameters, cache, imgvector, label_train)
        parameters = update_para(parameters, grads, learning_rate=ttt)
        grads["dW1"] = 0
        grads["dW2"] = 0
        grads["db1"] = 0
        grads["db2"] = 0
        # if i%1000==0:
        # pass
        print("cost after iteration %i:" % (i))
        print(costl)
    # print("ii de value: ",ii/50000.)
    # print('parameters',parameters["W1"],parameters["W2"],parameters["b1"],parameters["b2"])      # plt.imshow(train_images[i], cmap='gray')
    # print("cost : ",costl)
    # plt.show()
    # 测试
    for i in range(10000):
        img_train = test_images[i]
        vector_image = narmalize_data(image2vector(img_train))
        label_trainx = test_labels[i]
        aa2, xxx = forward_propagation(vector_image, parameters)
        predict_value = softmax(aa2)
        if predict_value == int(label_trainx):
            ii = ii + 1
        # print("the real value is: ",label_trainx)
        # print("the value of our prediction is: ",predict_value)
    print("正确率为：", ii / 100, "%")
