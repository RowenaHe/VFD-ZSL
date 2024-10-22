# ZhijiangLab Cup competition：zero-shot learning competition
# Team: ZJUAI
# Code function：Define functions to create the triplet loss with online triplet mining
# Reference paper: 《Discriminative Learning of Latent Features for Zero-Shot Recognition》

import torch
#import tensorflow as tf


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    temp = torch.transpose(embeddings,0,1)

    dot_product = torch.matmul(embeddings, temp)# 这一句 80 个核

    # 此处向上 80+ 核
    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)

    square_norm = torch.diag(dot_product)#tf.diag_part
    # 此处向上 80+ 核

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = torch.unsqueeze(square_norm, 1) - 2.0 * dot_product + torch.unsqueeze(square_norm, 0)#tf.expand_dims
    # 此处向上 80+ 核
    
    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.maximum(distances, torch.tensor(0.0))
    # 此处向上 80+ 核
    
    #print(distances.shape)
    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        #print(torch.eq(distances, torch.tensor(0.0)).type())
        mask = torch.eq(distances, torch.tensor(0.0)).type(torch.FloatTensor).to(distances.device)#tf.to_float
        #print('=========', distances.device)
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)
    
    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.shape[0]).type(torch.bool).to(labels.device)
    indices_not_equal = torch.logical_not(indices_equal)
    #print(labels.device)
    #print(indices_equal.shape, indices_not_equal.device)
    #print('label: ', labels.shape)
    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    #print(labels_equal.device)
    # Combine the two masks
    mask = torch.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))

    mask = torch.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.shape[0]).type(torch.BoolTensor).to(labels.device)#tf.cast( , tf.bool)
    indices_not_equal = torch.logical_not(indices_equal)
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)#tf.expand_dims
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)
    # 此处向上 1 核
    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    # print('=========',i_equal_j.device)
    valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))
    # 此处向上 1 核
    # Combine the two masks
    mask = torch.logical_and(distinct_indices, valid_labels)
    # print('mask:',mask)
    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    # 直接计算得到所有 embeddings 之间的距离，batch_size * batch_size 形状
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    # 此处向上 1 核

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)
    # 此处向上 1 核

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    # 广播机制将最后两维的 (1,bs) 和 (bs,1) 都扩成 (bs,bs)
    
    # 位置 (i,j,k) 上的值为 d(i,j) - d(i,k)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    # print(triplet_loss.shape)

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    # 只留 i,j 标签一样，i,k 标签不一样的为1，其她位置为 0
    mask = _get_triplet_mask(labels)
    # 此处向上 1 核

    mask = mask.to(triplet_loss.device)#.type(torch.FloatTensor)
    # print(mask.shape)
    #print('=======', triplet_loss.device, mask.device)

    triplet_loss = triplet_loss.mul(mask) #tf.multiply(mask, triplet_loss)
    # print(triplet_loss.device)
    # 此处向上 1 核

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.maximum(triplet_loss, torch.tensor(0.0))

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = torch.gt(triplet_loss, 1e-16)#.type(torch.FloatTensor) #tf.to_float
    num_positive_triplets = torch.sum(valid_triplets)#tf.reduce_sum
    num_valid_triplets = torch.sum(mask)
    # 此处向上 80+ 核

    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets
    

def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = mask_anchor_positive.type(torch.float32)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = torch.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)
    #torch.summary.scalar("hardest_positive_dist", torch.mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = mask_anchor_negative.type(torch.float32)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
    #print('===', mask_anchor_negative.shape, )
    #print(max_anchor_negative_dist, )
    #print(pairwise_dist.shape)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)
    #torch.summary.scalar("hardest_negative_dist", torch.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.maximum(hardest_positive_dist - hardest_negative_dist + margin, torch.tensor(0.0))

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss
