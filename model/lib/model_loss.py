# create custom loss function for training

import keras.backend as K

def audio_discriminate_loss(gamma=0.1,num_speaker=2):
    def loss_func(S_true,S_pred,gamma=gamma,num_speaker=num_speaker):
        sum = 0
        for i in range(num_speaker):
            sum += K.sum(K.flatten((K.square(S_true[:,:,:,i]-S_pred[:,:,:,i]))))
            for j in range(num_speaker):
                if i != j:
                    sum -= gamma*K.sum(K.flatten((K.square(S_true[:,:,:,i]-S_pred[:,:,:,j]))))

        loss = sum / (num_speaker*298*257*2)
        return loss
    return loss_func


def audio_discriminate_loss2(gamma=0.1,beta = 2*0.1,num_speaker=2):
    def loss_func(S_true,S_pred,gamma=gamma,beta=beta,num_speaker=num_speaker):
        sum_mtr = K.zeros_like(S_true[:,:,:,:,0])
        for i in range(num_speaker):
            sum_mtr += K.square(S_true[:,:,:,:,i]-S_pred[:,:,:,:,i])
            for j in range(num_speaker):
                if i != j:
                    sum_mtr -= gamma*(K.square(S_true[:,:,:,:,i]-S_pred[:,:,:,:,j]))

        for i in range(num_speaker):
            for j in range(i+1,num_speaker):
                #sum_mtr -= beta*K.square(S_pred[:,:,:,i]-S_pred[:,:,:,j])
                #sum_mtr += beta*K.square(S_true[:,:,:,:,i]-S_true[:,:,:,:,j])
                pass
        #sum = K.sum(K.maximum(K.flatten(sum_mtr),0))

        loss = K.mean(K.flatten(sum_mtr))

        return loss
    return loss_func




