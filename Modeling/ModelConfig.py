# configuration about CPN model
# you can config these parameters to train CPN
epoch=None # recommendation is 800
id=None # index of cross validation part. Random split dataset if it is None. Recommendation is 1 to 5.
lr=None #learning rate, recommendation is 0.12，with larger batchsize
batchsize=None #batchsize while training in a epoch, recommendation is 2^x, but we set it as 650 with previous experiences
####################
epoch_div=None # recommendation is 50
epoch_mt=None # recommendation is 100
lr_dec=None # recommendation is 0.8 to 0.99
# above is scheduler configuration about CPN
# if epoch_div % 50 == 0 and epoch != 0 and epoch>=epoch_mt: lr=lr*lr_dec
#####################