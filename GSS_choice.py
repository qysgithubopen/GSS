import torch
import numpy as np

def singel_teacher(pre1, pre2, pre3, pre4, current_step, max_step, pre1_logits, pre2_logits, pre3_logits, pre4_logits, T1, T2):


    pre1_mask = pre1 > T1



    pre_logits = pre1_logits.clone()

    pre_logits[~pre1_mask] = 0



    pre2_temp = pre2.clone()
    pre2_temp[pre1_mask] = 0
    pre2_logits_temp = pre2_logits.clone()
    pre2_logits_temp[~pre1_mask] = 0

    pre3_temp = pre3.clone()
    pre3_temp[pre1_mask] = 0
    pre3_logits_temp = pre3_logits.clone()
    pre3_logits_temp[~pre1_mask] = 0

    pre4_temp = pre4.clone()
    pre4_temp[pre1_mask] = 0
    pre4_logits_temp = pre4_logits.clone()
    pre4_logits_temp[~pre1_mask] = 0


    pre2_mask = pre2_temp > T2
    pre3_mask = pre3_temp > T2
    pre4_mask = pre4_temp > T2



    low_weight_mask = pre4_mask & (pre2_mask & pre3_mask)

    chose_mask = ~low_weight_mask
    pre2_temp[chose_mask] = 0
    pre3_temp[chose_mask] = 0
    pre4_temp[chose_mask] = 0

    pre2_logits_temp[chose_mask] = 0
    pre3_logits_temp[chose_mask] = 0
    pre4_logits_temp[chose_mask] = 0



    pre2_3_4_logits = (pre2_logits_temp + pre3_logits_temp + pre4_logits_temp)/3



    pre1_logits[pre1_mask] = 0
    pre2_logits[pre1_mask] = 0
    pre3_logits[pre1_mask] = 0
    pre4_logits[pre1_mask] = 0

    pre1_logits[low_weight_mask] = 0
    pre2_logits[low_weight_mask] = 0
    pre3_logits[low_weight_mask] = 0
    pre4_logits[low_weight_mask] = 0



    pre_min_logits, _ = torch.min(torch.cat([pre1_logits, pre2_logits, pre3_logits, pre4_logits],dim=1), dim=1, keepdim=True)


    pre_logits = pre_logits + pre2_3_4_logits + pre_min_logits

    return pre_logits


def double_teacher(pre1, pre2, pre3, pre4, current_step, max_step, pre1_logits, pre2_logits, pre3_logits, pre4_logits, T1, T2, P):

    pre1_mask = pre1 > T1
    pre2_mask = pre2 > T1

    pre_domain_mask = pre1_mask | pre2_mask



    pre1_temp = pre1.clone()
    pre2_temp = pre2.clone()

    pre1_logits_temp = pre1_logits.clone()
    pre2_logits_temp = pre2_logits.clone()

    pre1_temp[~pre_domain_mask] = 0

    pre2_temp[~pre_domain_mask] = 0

    pre1_logits_temp[~pre_domain_mask] = 0
    pre2_logits_temp[~pre_domain_mask] = 0

    pre_domain_mask1 = pre1_mask & pre2_mask

    pre1_temp1 = pre1.clone()
    pre2_temp1 = pre2.clone()

    pre1_logits_temp1 = pre1_logits.clone()
    pre2_logits_temp1 = pre2_logits.clone()

    pre1_temp1[~pre_domain_mask1] = 0

    pre2_temp1[~pre_domain_mask1] = 0

    pre1_logits_temp1[~pre_domain_mask1] = 0
    pre2_logits_temp1[~pre_domain_mask1] = 0



    max1_21, _ = torch.max(torch.cat([pre1_logits_temp1, pre2_logits_temp1],dim=1), dim=1, keepdim=True)

    pre_domain1 =  max1_21

    pre_domain_mask2 = pre_domain_mask ^ pre_domain_mask1

    pre1_temp2 = pre1.clone()
    pre2_temp2 = pre2.clone()

    pre1_logits_temp2 = pre1_logits.clone()
    pre2_logits_temp2 = pre2_logits.clone()
    pre1_temp2[~pre_domain_mask2] = 0
    pre2_temp2[~pre_domain_mask2] = 0
    pre1_logits_temp2[~pre_domain_mask] = 0
    pre2_logits_temp2[~pre_domain_mask] = 0


    max1_22, _ = torch.max(torch.cat([pre1_logits_temp2, pre2_logits_temp2],dim=1), dim=1, keepdim=True)
    min1_22, _ = torch.min(torch.cat([pre1_logits_temp2, pre2_logits_temp2],dim=1), dim=1, keepdim=True)
    pre_domain2 =  max1_22 - P*(max1_22 - min1_22)

    pre3_temp = pre3.clone()
    pre3_temp[pre_domain_mask] = 0
    pre3_logits_temp = pre3_logits.clone()
    pre3_logits_temp[pre_domain_mask] = 0


    pre4_temp = pre4.clone()
    pre4_temp[pre_domain_mask] = 0
    pre4_logits_temp = pre4_logits.clone()
    pre4_logits_temp[pre_domain_mask] = 0

    pre3_mask = pre3_temp > T2
    pre4_mask = pre4_temp > T2

    low_weight_mask = pre4_mask & pre3_mask

    chose_mask = ~low_weight_mask
    pre3_temp[chose_mask] = 0
    pre4_temp[chose_mask] = 0

    pre3_logits_temp[chose_mask] = 0
    pre4_logits_temp[chose_mask] = 0


    pre3_4 = (pre3_logits_temp + pre4_logits_temp)/2

    pre1_logits[pre_domain_mask] = 0
    pre2_logits[pre_domain_mask] = 0
    pre3_logits[pre_domain_mask] = 0
    pre4_logits[pre_domain_mask] = 0

    pre1_logits[low_weight_mask] = 0
    pre2_logits[low_weight_mask] = 0
    pre3_logits[low_weight_mask] = 0
    pre4_logits[low_weight_mask] = 0

    pre_min, _ = torch.min(torch.cat([pre1_logits, pre2_logits, pre3_logits, pre4_logits],dim=1), dim=1, keepdim=True)

    pre = pre_domain1 + pre_domain2 + pre3_4 + pre_min

    return pre


def singel_teacher_np(pre1, pre2, pre3, pre4, current_step, max_step, pre1_logits, pre2_logits, pre3_logits, pre4_logits, T1, T2):

    pre1_mask = pre1 > T1
    pre_logits = pre1_logits.clone()
    pre_logits[np.logical_not(pre1_mask)] = 0

    pre2_temp = pre2.clone()
    pre2_temp[pre1_mask] = 0
    pre2_logits_temp = pre2_logits.clone()
    pre2_logits_temp[np.logical_not(pre1_mask)] = 0

    pre3_temp = pre3.clone()
    pre3_temp[pre1_mask] = 0
    pre3_logits_temp = pre3_logits.clone()
    pre3_logits_temp[np.logical_not(pre1_mask)] = 0

    pre4_temp = pre4.clone()
    pre4_temp[pre1_mask] = 0
    pre4_logits_temp = pre4_logits.clone()
    pre4_logits_temp[np.logical_not(pre1_mask)] = 0

    pre2_mask = pre2_temp > T2
    pre3_mask = pre3_temp > T2
    pre4_mask = pre4_temp > T2



    low_weight_mask = np.logical_and(pre4_mask, np.logical_and(pre2_mask, pre3_mask))

    chose_mask = np.logical_not(low_weight_mask)
    pre2_temp[chose_mask] = 0
    pre3_temp[chose_mask] = 0
    pre4_temp[chose_mask] = 0

    pre2_logits_temp[chose_mask] = 0
    pre3_logits_temp[chose_mask] = 0
    pre4_logits_temp[chose_mask] = 0

    pre2_3_4_logits = (pre2_logits_temp + pre3_logits_temp + pre4_logits_temp)/3



    pre1_logits[pre1_mask] = 0
    pre2_logits[pre1_mask] = 0
    pre3_logits[pre1_mask] = 0
    pre4_logits[pre1_mask] = 0

    pre1_logits[low_weight_mask] = 0
    pre2_logits[low_weight_mask] = 0
    pre3_logits[low_weight_mask] = 0
    pre4_logits[low_weight_mask] = 0



    pre_min_logits, _ = np.min(np.concatenate([pre1_logits, pre2_logits, pre3_logits, pre4_logits],dim=1), dim=1, keepdim=True)


    pre_logits = pre_logits + pre2_3_4_logits + pre_min_logits

    return pre_logits


def double_teacher_np(pre1, pre2, pre3, pre4, current_step, max_step, pre1_logits, pre2_logits, pre3_logits, pre4_logits, T1, T2, P):

    pre1_mask = pre1 > T1
    pre2_mask = pre2 > T1

    pre_domain_mask = np.logical_or(pre1_mask, pre2_mask)



    pre1_temp = pre1.clone()
    pre2_temp = pre2.clone()

    pre1_logits_temp = pre1_logits.clone()
    pre2_logits_temp = pre2_logits.clone()

    pre1_temp[np.logical_not(pre_domain_mask)] = 0

    pre2_temp[np.logical_not(pre_domain_mask)] = 0

    pre1_logits_temp[np.logical_not(pre_domain_mask)] = 0
    pre2_logits_temp[np.logical_not(pre_domain_mask)] = 0

    # ---------------------------------------------------------------------------
    pre_domain_mask1 = np.logical_and(pre1_mask, pre2_mask)

    pre1_temp1 = pre1.clone()
    pre2_temp1 = pre2.clone()

    pre1_logits_temp1 = pre1_logits.clone()
    pre2_logits_temp1 = pre2_logits.clone()

    pre1_temp1[np.logical_not(pre_domain_mask1)] = 0

    pre2_temp1[np.logical_not(pre_domain_mask1)] = 0

    pre1_logits_temp1[np.logical_not(pre_domain_mask1)] = 0
    pre2_logits_temp1[np.logical_not(pre_domain_mask1)] = 0



    max1_21, _ = np.max(np.concatenate([pre1_logits_temp1, pre2_logits_temp1],dim=1), dim=1, keepdim=True)

    pre_domain1 =  max1_21

    pre_domain_mask2 = np.logical_xor(pre_domain_mask, pre_domain_mask1)

    pre1_temp2 = pre1.clone()
    pre2_temp2 = pre2.clone()

    pre1_logits_temp2 = pre1_logits.clone()
    pre2_logits_temp2 = pre2_logits.clone()

    pre1_temp2[np.logical_not(pre_domain_mask2)] = 0

    pre2_temp2[np.logical_not(pre_domain_mask2)] = 0

    pre1_logits_temp2[np.logical_not(pre_domain_mask)] = 0
    pre2_logits_temp2[np.logical_not(pre_domain_mask)] = 0




    max1_22, _ = np.max(np.concatenate([pre1_logits_temp2, pre2_logits_temp2],dim=1), dim=1, keepdim=True)
    min1_22, _ = np.min(np.concatenate([pre1_logits_temp2, pre2_logits_temp2],dim=1), dim=1, keepdim=True)
    pre_domain2 =  max1_22 - P*(max1_22 - min1_22)







    pre3_temp = pre3.clone()
    pre3_temp[pre_domain_mask] = 0
    pre3_logits_temp = pre3_logits.clone()
    pre3_logits_temp[pre_domain_mask] = 0


    pre4_temp = pre4.clone()
    pre4_temp[pre_domain_mask] = 0
    pre4_logits_temp = pre4_logits.clone()
    pre4_logits_temp[pre_domain_mask] = 0

    pre3_mask = pre3_temp > T2
    pre4_mask = pre4_temp > T2



    low_weight_mask = np.logical_and(pre4_mask, pre3_mask)



    chose_mask = np.logical_not(low_weight_mask)
    pre3_temp[chose_mask] = 0
    pre4_temp[chose_mask] = 0
    pre3_logits_temp[chose_mask] = 0
    pre4_logits_temp[chose_mask] = 0






    pre3_4 = (pre3_logits_temp + pre4_logits_temp)/2

    pre1_logits[pre_domain_mask] = 0
    pre2_logits[pre_domain_mask] = 0
    pre3_logits[pre_domain_mask] = 0
    pre4_logits[pre_domain_mask] = 0

    pre1_logits[low_weight_mask] = 0
    pre2_logits[low_weight_mask] = 0
    pre3_logits[low_weight_mask] = 0
    pre4_logits[low_weight_mask] = 0




    pre_min, _ = np.min(np.concatenate([pre1_logits, pre2_logits, pre3_logits, pre4_logits],dim=1), dim=1, keepdim=True)



    pre = pre_domain1 + pre_domain2 + pre3_4 + pre_min

    return pre
