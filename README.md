# yomamai 


'''
__GENERAL__
https://arxiv.org/abs/1206.5533
https://stats.stackexchange.com/questions/242004/why-do-neural-network-researchers-care-about-epochs
__MISC__
https://medium.com/jim-fleming/notes-on-the-numerai-ml-competition-14e3d42c19f3
https://github.com/jsilter/parametric_tsne
https://www.kaggle.com/timevans/rf-xgboost-keras
https://www.kaggle.com/hsperr/finding-ensamble-weights

https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

CUDA_VISIBLE_DEVICES='1' 
CUDA_VISIBLE_DEVICES='1' jupyter notebook
tensorboard --logdir models/moddisentablegan_file/log
'''

* create ethereum wallet (https://medium.com/tokenclub/how-to-create-your-own-ethereum-wallet-using-myetherwallet-fcb494e1a053 or https://github.com/ethereum/go-ethereum/wiki/Managing-your-accounts).
* install and create MetaMask account in browser (https://metamask.io/).
* import eth wallet using privatekey or keystore to MetaMask.
* from numer.ai withdraw USD to ethereum wallet, which is stored as ETH.
* convert ETH to NMR on an exchange, I used shapeshift.io in conjunction with MetaMask, at this point, you can then direct the NMR to the same ethereum wallet or to your numer.ai account.
* monitor the transaction process with https://etherscan.io
* confirm transaction by checking the wallet in https://etherscan.io, under Misc, there should be a Token Tracker that shows you the available NMR you have.
* to send/receive NMR, you can use myetherwallet (https://www.myetherwallet.com more instructions: http://www.tokenverse.com/blog/how-to-transfer-your-mbrs-using-metamask-and-myetherwallet-ethplorer-explanation-step-by-step/)
