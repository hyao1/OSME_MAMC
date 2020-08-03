import tensorflow as tf

def mamc_loss(targets, input_fc):
    """
        input_fc是class_num*attention_num个注意力向量,shape(bitch_size, attention_num*1024)
        target是每个向量对应的类别，用长为class_num的向量表示
    """

    attention_num = 2
    bitch_size = 8
    dim = 1024
    n = bitch_size * attention_num  # 向量个数

    # 输入向量变为n个dim维向量，然后和转置相乘
    input_fc = tf.reshape(input_fc, (n, dim))
    input_fc = tf.cast(input_fc, dtype=tf.float64)
    prod = tf.matmul(input_fc, input_fc, transpose_b=True)

    # targets张量
    targets = tf.where(tf.equal(targets, tf.ones_like(targets)))[:, 1]
    targets = tf.reshape(tf.tile(tf.reshape(targets, [bitch_size, 1]), [1, attention_num]), [1, n])
    targets = tf.tile(targets, (n, 1))

    # attention张量
    attention = tf.tile(tf.range(0, attention_num), [bitch_size])
    attention = tf.tile(tf.reshape(attention, [1, n]), [n, 1])

    same_class_mask = tf.equal(targets, tf.transpose(targets))
    same_atten_mask = tf.equal(attention, tf.transpose(attention))
    s_sasc = same_class_mask & same_atten_mask
    s_sadc = (~same_class_mask) & same_atten_mask
    s_dasc = same_class_mask & (~same_atten_mask)
    s_dadc = (~same_class_mask) & (~same_atten_mask)

    loss_sasc = tf.constant(0, dtype=tf.float64)
    loss_sadc = tf.constant(0, dtype=tf.float64)
    loss_dasc = tf.constant(0, dtype=tf.float64)

    for i in range(n):
        # print(session.run(prod[i]))
        # loss_sasc
        pos = tf.gather(prod[i], tf.where(tf.reshape(s_sasc[i], [-1])))
        neg = tf.gather(prod[i], tf.where(tf.reshape([s_sadc[i] | s_dasc[i] | s_dadc[i]], [-1])))
        neg = tf.transpose(neg)

        n_pos = tf.shape(pos)[0]
        n_neg = tf.shape(neg)[1]

        pos = tf.tile(pos, [1, n_neg])
        neg = tf.tile(neg, [n_pos, 1])

        sasc = neg - pos
        loss_sasc = loss_sasc + tf.reduce_sum(tf.log(1 + tf.reduce_sum(tf.exp(sasc), axis=1)))

        # loss_sadc
        pos = tf.gather(prod[i], tf.where(tf.reshape(s_sadc[i], [-1])))
        neg = tf.gather(prod[i], tf.where(tf.reshape(s_dadc[i], [-1])))
        neg = tf.transpose(neg)

        n_pos = tf.shape(pos)[0]
        n_neg = tf.shape(neg)[1]

        pos = tf.tile(pos, [1, n_neg])
        neg = tf.tile(neg, [n_pos, 1])

        sadc = neg - pos
        loss_sadc = loss_sadc + tf.reduce_sum(tf.log(1 + tf.reduce_sum(tf.exp(sadc), axis=1)))

        # loss_dasc
        pos = tf.gather(prod[i], tf.where(tf.reshape(s_dasc[i], [-1])))
        neg = tf.gather(prod[i], tf.where(tf.reshape(s_dadc[i], [-1])))
        neg = tf.transpose(neg)

        n_pos = tf.shape(pos)[0]
        n_neg = tf.shape(neg)[1]

        pos = tf.tile(pos, [1, n_neg])
        neg = tf.tile(neg, [n_pos, 1])

        dasc = neg - pos
        loss_dasc = loss_dasc + tf.reduce_sum(tf.log(1 + tf.reduce_sum(tf.exp(dasc), axis=1)))

    loss = (loss_sasc + loss_sadc + loss_dasc) / tf.cast(tf.shape(input_fc)[0], dtype=tf.float64)
    return tf.cast(loss, dtype=tf.float32)


def softmax_loss(targets, input_predict):
    loss = input_predict * targets
    loss = tf.reduce_sum(-tf.log(tf.reduce_sum(loss, axis=1)))  # /tf.cast(tf.shape(targets)[0], dtype=tf.float32)
    return loss


