import pandas as pd
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
from scipy import signal
from astropy.stats import median_absolute_deviation
from scipy.stats import iqr
from scipy.stats import entropy
from pylab import plot, log10, linspace, axis
from spectrum import *
import pandas as pd
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
from scipy import signal
from astropy.stats import median_absolute_deviation
from scipy.stats import iqr
from scipy.stats import entropy
from pylab import plot, log10, linspace, axis
from spectrum import *
# def generrate_signal(dir):
#     df=pd.read_csv(dir,header=None,sep=',')#dir='Gyro_t.csv'
#     L=len(df[2][1:])
#     accsig,gyrosig,gravitysig=[],[],[]
#     for i in range(1,4):
#         tem=[]
#         for j in range(1,len(df[i])):
#             tem.append(float(df[i][j]))
#             #+random.gauss(mu,sigma)
#         gravitysig.append(tem)
#     for i in range(4,7):
#         tem=[]
#         for j in range(1,len(df[i])):
#             tem.append(float(df[i][j]))
#             #+random.gauss(mu,sigma)
#         gyrosig.append(tem)
#     for i in range(7,10):
#         tem=[]
#         for j in range(1,len(df[i])):
#             tem.append(float(df[i][j]))
#             #+random.gauss(mu,sigma)
#         accsig.append(tem)
#     labels=df[10][1:]
#     x = np.linspace(0, 1, len(df[1][1:]))
#     return accsig,gyrosig,gravitysig,x,labels
#     #
#     # fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(5, 1, sharex=True)
#     # ax1.plot(x, sig[0])
#     # ax1.set_title('sinusoids')
#     # ax1.axis([0, 1, -1, 1])
#
#
# ####范数
# def Euclidean_norm(x,y,z):
#     return (x**2+y**2+z**2)**(1/3)
# def CalMag(sigs):
#     Mag=[]
#     for i in range(len(sigs[0])):
#         Mag.append(Euclidean_norm(sigs[0][i],sigs[1][i],sigs[2][i]))
#     return Mag
# # Mag=CalMag(sig)
# # ax5.plot(x, Mag)
# # ax5.set_title('Magsignal')
# # ax5.axis([0, 1, -1, 1])
#
#
# ########Wn=2*截止频率/采样频率
# #########中值滤波+三阶巴特沃斯滤波器20hz去噪###20*2/100hz
# def MidFilter(sigs):
#     mid = signal.medfilt(sigs)
#     # sos =signal.butter(3, 0.4, 'lp', output='sos')
#     # filtered = signal.sosfilt(sos, mid,axis=0)
#     # b, a = signal.butter(3, 0.4, 'lowpass')
#     # filtered = signal.filtfilt(b, a, mid,axis=1)#data为要过滤的信号
#     sos20 = signal.butter(3, 0.4, 'lp', fs=100, output='sos')
#     filtered= signal.sosfilt(sos20, sigs)
#     return filtered
# # ax2.plot(x, filtered)
# # ax2.set_title('After 20 Hz low-pass filter')
# # ax2.axis([0, 1, -1, 1])
# # ax2.set_xlabel('Time [seconds]')
#
#
# ######三阶巴特沃斯滤波器 0.3hz分离重力加速度和身体加速度
#
# #####tbodyacc##########
# def splitGandB(sigs):
#     sos2 =signal.butter(3, 0.006, 'lp', fs=100, output='sos')
#     filtered1 = signal.sosfilt(sos2, sigs,axis=0)
#     # ax3.plot(x, filtered1)
#     # ax3.set_title('After 0.3 Hz low-pass filter')
#     # ax3.axis([0, 1, -0.2, 0.2])
#     # ax3.set_xlabel('Time [seconds]')
#     #
#
#     sos3=signal.butter(3, 0.006, 'hp', fs=100, output='sos')
#     filtered2 = signal.sosfilt(sos3, sigs)
#     # ax4.plot(x, filtered2)
#     # ax4.set_title('After 0.3 Hz high-pass filter')
#     # ax4.axis([0, 1, -0.2, 0.2])
#     # ax4.set_xlabel('Time [seconds]')
#     # plt.tight_layout()
#     # plt.show()
#     return filtered1,filtered2
#
#
# ##########Jerk##############3
# def jerk(sigs):
#     Jerksig=[]
#     for i in range(1,len(sigs)):
#         Jerksig.append((sigs[i]-sigs[i-1])/0.01)
#     # print(Jerksig)
#     return Jerksig
#
# #######################
# #####傅里叶##########f
#
# def fly(sigs,x,fs):#x----linspace
#     fs=2000
#     yy =fft(sigs,fs)
#     yf=abs(yy)                # 取绝对值
#     yf1=abs(yy/len(x) )          #归一化处理
#     yf2 = yf1[1:int(len(x)/2)]  #取一半区间
#     dc=yf1[0]
#     xf = np.arange(0,1.,1/fs)
#     xf.tolist()# 频率
#     xf1 = xf
#     xf2 = xf[1:int(len(x)/2)]  #取一半区间
#     return dc,xf2,yf2
#
#
# # plt.subplot(221)
# # plt.plot(x[0:500],sig[0][0:500])
# # plt.title('Original wave')
# #
# # plt.subplot(222)
# # plt.plot(xf,yf,'r')
# # plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')
# #
# # plt.subplot(223)
# # plt.plot(xf1,yf1,'g')
# # plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')
# #
# # plt.subplot(224)
# # plt.plot(xf2,yf2,'b')
# # plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')
# #
# #
# # plt.show()
# # print(dft_a)
# # plt.plot(dft_a)
# # plt.grid(True)
# # plt.xlim(0, 15)
# # plt.show()
# #
# # print(df)
# # print(df.shape)
# # print(df.info())
#
#
#
#
# #mean（）：平均值
# def Meansig(sigs):
#     m=np.mean(sigs)
#     return m
# #std（）：标准偏差
# def std(sigs):
#     sig_std = np.std(sigs, ddof=1)
#     return sig_std
# #中位c查
# def MAD(sigs):
#     mads=median_absolute_deviation(sigs)
#     return mads
# # max（）：数组中的最大值
# def sigMAX(sigs):
#     return max(sigs)
# # min（）：数组中的最小值
# def sigMIN(sigs):
#     return min(sigs)
# # sma（）：信号幅度区域###3-dims
# def sigSMA(sigs):
#     all=0.
#     if len(sigs)==1:
#         for i in sigs:
#             all+=abs(i)
#         all/=len(sigs)
#     if len(sigs)==3:
#         for i in range(len(sigs[0])):
#             all+=abs(sigs[0][i])+abs(sigs[1][i])+abs(sigs[2][i])
#         all/=len(sigs[0])
#     return all
# # energy（）：能量度量平方和除以数量。####1 dim
# #####归一化#######################################################################
# #####
# # %% 将数据归一化到[a,b]区间的方法
# # a=0.1;
# # b=0.5;
# # Ymax=max(y);%计算最大值
# # Ymin=min(y);%计算最小值
# # k=(b-a)/(Ymax-Ymin);
# # norY=a+k*(y-Ymin);
# ####################################################################################3
# def norm(sigs):
#     newsig=sigs
#     Max=max(sigs)
#     Min=min(sigs)
#     a=-1
#     b=1
#     k=(b-a)/(Max-Min)
#     for i in range(len(newsig)):
#         newsig[i]=a+k*(newsig[i]-Min)
#     return newsig
# def energy(sigs):
#
#     en=0.
#     for i in sigs:
#         en+=i**2
#     en/=len(sigs)
#     return en
#
# # iqr（）：四分位数范围
# def iqrs(sigs):
#     return iqr(sigs)
#
#
# # entropy（）：信号熵
# def sigEntropy(sigs):
#     for i in range(len(sigs)):
#         if sigs[i] < 0:
#             sigs[i] = -sigs[i]
#     return entropy(sigs,base=2)
# # arCoeff（）：Burg阶等于4的自回归系数
# def arCoeff(sigs):
#     AR, P, k = arburg(sigs, 4)
#     AR=abs(AR)
#     # print('cor!!!!!!!!!!!!!!!!',AR[0],AR[1],AR[2],AR[3])
#     return [AR[0],AR[1],AR[2],AR[3]]
# # related（）：两个信号之间的相关系数
# def corr(sigs):
#     a,b,c=sigs[0],sigs[1],sigs[2]
#     x1=np.corrcoef(a,b)[1][0]
#     x2=np.corrcoef(a,c)[1][0]
#     x3=np.corrcoef(b,c)[1][0]
#     return [x1,x2,x3]
#
# # maxInds（）：幅度最大的频率分量的索引###对FFT后
# def maxinds(sigs,fs):
#     if (type(sigs).__name__ != 'list'):
#         sig=sigs.tolist()
#     else:
#         sig=sigs
#     idx,m= sig.index(max(sig)),max(sig)
#     return (0.5-(idx/fs))*m
# # meanFreq（）：获得平均频率的频率分量的加权平均值###FFT后
#
# def meanFreq(sigs):
#     sum=0.
#     for i in sigs:
#         sum+=i
#     mean=sum/len(sigs)
#     tot=0.
#     for i in sigs:
#         tot+=i*i/mean
#     return tot/len(sigs)
#
#
# # kurtosis（）：频域信号的峰度
# # skewness（）：频域信号的偏斜度
# #FFT
# def SkewAndKur(sigs):
#     sk=stats.skew(sigs, axis=0, bias=True)
#     ku = stats.kurtosis(sigs)
#     return [abs(sk),abs(ku)]
#
#
# # bandsEnergy（）：每个窗口的FFT的64个bin内的频率间隔的能量。
# def bandEnergy(sigs,fs):  ####FFT
#     # print(len(sig))
#     inter = fs//8  #
#     res = []
#     for i in range(8):
#         x = 0
#         # for j in range(8):
#             # for i in range(inter):
#             #     x += sig[j * inter + i] ** 2
#             # res.append(x / inter)
#         for j in sigs[i*inter:inter*(i+1)]:
#             x+=j
#         res.append(x/inter)
#     res.append((res[0]+res[1])/2)
#     res.append((res[2]+res[3])/2)
#     res.append((res[4]+res[5])/2)
#     res.append((res[6] + res[7])/2)
#     res.append((res[8]+res[9])/2)
#     res.append((res[10] + res[11]) / 2)
#     return res###14个数
#
# # angle（）：矢量之间的角度。
# def angle(x,y):
#     if x=='X':
#         x=[1,0,0]
#     elif x=='Y':
#         x=[0,1,0]
#     elif x=='Z':
#         x=[0,0,1]
#     absx,absy,xy=0,0,0
#     for i in range(3):
#         absx+=x[i]**2
#         absy+=y[i]**2
#         xy+=x[i]*y[i]
#     absx=absx**0.5
#     absy=absy**0.5
#     angle=xy/(absy*absx)
#     return angle
# ############MEAN ###############################################
# ##Additional vectors obtained by averaging the signals in a ####
# # signal window sample. These are used on the angle() variable:##
# #################################################################
# def calMEAN(sigs):#signal with 3 dims
#     xyz=[]
#     for i in range(3):
#         xyz.append(np.mean(sigs[i]))
#     return xyz
#
#
#
#
# # def GyroAndAcc(dir):
# #     Gyrosig,x1=generrate_signal(gyrodir)
# #     Accsig,x2=generrate_signal(accdir)
# #     return Gyrosig,Accsig,x2,x1
# def ALLFeature(fs,accsig_idx,gyrosig_idx,gravitysig_idx,xacc):
#     FEATURE=[]
#     # tBodyAcc, tgyro, tGravity,xacc,labels=generrate_signal(dir)
#     xgro=xacc
#     # tgyro,tAcc,xacc,xgro=GyroAndAcc(dir)#acc,gyro
#     # for i in range(3):
#     #     tgyro[i]=MidFilter(tgyro[i])
#     #     tAcc[i]=MidFilter(tAcc[i])
#     # tBodyAcc - XYZ
#     # tGravityAcc - XYZ
#     tgyro,tBodyAcc,tGravity=gyrosig_idx,accsig_idx,gravitysig_idx
#     tBodyAccJerk,tBodyGyroJerk,fbodyacc,fbodyaccjerk,fbodygyro,fbodygyrojerk=[],[],[],[],[],[]
#     for i in range(3):
#         tgyro[i] = MidFilter(tgyro[i])
#         # tAcc[i] = MidFilter(tAcc[i])
#         # tBodyAcc1,tGravity1=splitGandB(tAcc[i])
#         tBodyAcc=MidFilter(tBodyAcc[i])
#         tGravity=MidFilter(tGravity[i])
#         tbodyAccjerk1=jerk(tBodyAcc[i])
#         tbodyGyroJerk1=jerk(tgyro[i])
#         tBodyAccJerk.append(tbodyAccjerk1)
#         tBodyGyroJerk.append(tbodyGyroJerk1)
#         ###FFT
#         # fBodyAcc - XYZ
#         # fBodyAccJerk - XYZ
#         # fBodyGyro - XYZ
#         fbodyaccDC,fbodyacc_x,fbodyacc_y=fly(tBodyAcc[i],xacc,fs)
#         fbodyaccjerkDC,fbodyaccjerk_x,fbodyaccjerk_y=fly(tbodyAccjerk1,xacc,fs)
#         fbodygyroDC,fbodygyro_x,fbodygyro_y=fly(tgyro[i],xgro,fs)
#         fbodygyrojerkDC, fbodygyrojerk_x, fbodygyrojerk_y = fly(tbodyAccjerk1,xgro,fs)
#         fbodygyrojerk.append(fbodygyrojerk_y)
#         fbodyacc.append(fbodyacc_y)
#         fbodyaccjerk.append(fbodyaccjerk_y)
#         fbodygyro.append(fbodygyro_y)
#
#
#
#     # tBodyAccMag
#     # tGravityAccMag
#     # tBodyAccJerkMag
#     # tBodyGyroMag
#     # tBodyGyroJerkMag
#     tBodyAccMag=CalMag(tBodyAcc)
#     tGravityAccMag=CalMag(tGravity)
#     tBodyAccJerkMag=CalMag(tBodyAccJerk)
#     tBodyGyroMag=CalMag(tgyro)
#     tBodyGyroJerkMag=CalMag(tBodyGyroJerk)
#
#     # fBodyAccMag
#     # fBodyAccJerkMag
#     # fBodyGyroMag
#     # fBodyGyroJerkMag
#     fBodyAccMag=CalMag(fbodyacc)
#     fBodyAccJerkMag=CalMag(fbodyaccjerk)
#     fBodyGyroMag=CalMag(fbodygyro)
#     # fbodygyrojerk=jerk(fbodygyro)
#     # print('*****************',fbodygyrojerk,'**********')
#     # print(fbodygyrojerk[1],'^^^^^^^^^^^^^^^^^^^^^^^^^')
#     # print(fbodygyrojerk[2],'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2')
#
#     fBodyGyroJerkMag=CalMag(fbodygyrojerk)
# ########################################################################
# ####  [tgyro, tBodyAcc, tGravity,tBodyAccJerk,tBodyGyroJerk,fbodyacc,fbodyaccjerk,fbodygyro,fbodygyrojerk,tBodyAccMag
#     # tGravityAccMag, tBodyAccJerkMag,tBodyGyroMag,tBodyGyroJerkMag,fBodyAccMag, fBodyAccJerkMag, fBodyGyroMag
# #### fBodyGyroJerkMag  ]
# ##############################################################################
#     TimefeatureList=[tBodyAcc,tGravity,tBodyAccJerk,tgyro,tBodyGyroJerk,tBodyAccMag, tGravityAccMag,tBodyAccJerkMag,tBodyGyroMag,tBodyGyroJerkMag]
#     FreqfeatureList=[fbodyacc,fbodyaccjerk,fbodygyro,fBodyAccMag,fBodyAccJerkMag,fBodyGyroMag,fBodyGyroJerkMag]
#
#     #################
#     # mean（）：平均值
#     # std（）：标准偏差
#     # mad（）：中值绝对偏差
#     # max（）：数组中的最大值
#     # min（）：数组中的最小值
#     # t    # sma（）：信号幅度区域
#     # energy（）：能量度量平方和除以数量。
#     # iqr（）：四分位数范围
#     # entropy（）：信号熵
#     # arCoeff（）：Burg阶等于4的自回归系数
#     # related（）：两个信号之间的相关系数
#     #####################
#
#     for i in TimefeatureList:
#         mean3,std3,mad3,min3,max3,sma1,energy3,iqr3,entropy3,arCoeff12,correction3,=[],[],[],[],[],[],[],[],[],[],[]
#         if len(i)==3:
#             for j in range(len(i)):
#                 mean3.append(Meansig(i[j]))
#                 std3.append(std(i[j]))
#                 mad3.append(MAD(i[j]))
#                 min3.append(sigMIN(i[j]))
#                 max3.append(sigMAX(i[j]))
#                 energy3.append(energy(i[j]))
#                 iqr3.append(iqrs(i[j]))
#                 entropy3.append(sigEntropy(i[j]))
#                 arCoeff12+=arCoeff(i[j])
#             correction3+=corr(i)
#             sma1.append(sigSMA(i))
#             # print('mean3',len(mean3),'mean3',len(std3),'mad3',len(mad3),'mad3',len(max3),'min3',len(min3),'sma1',len(sma1),'energy3',len(energy3),'iqr',len(iqr3),'entropy',len(entropy3),'arCoeff12',len(arCoeff12),'correction3',len(correction3))
#             FEATURE += mean3 + std3 + mad3 + max3 + min3 + sma1 + energy3 + iqr3 + entropy3 + arCoeff12 + correction3
#             # print(len(FEATURE))
#         else:
#             mean3.append(Meansig(i))
#             std3.append(std(i))
#             mad3.append(MAD(i))
#             min3.append(sigMIN(i))
#             max3.append(sigMAX(i))
#             sma1.append(sigSMA(i))
#             energy3.append(energy(i))
#             iqr3.append(iqrs(i))
#             entropy3.append(sigEntropy(i))
#             arCoeff12 += arCoeff(i)
#             # print('mean31',len(mean3),'mean31',len(std3),'mad3',len(mad3),'mad3',len(max3),'min3',len(min3),'sma1',len(sma1),'energy3',len(energy3),'iqr',len(iqr3),'entropy',len(entropy3),'arCoeff12',len(arCoeff12))
#             FEATURE += mean3 + std3 + mad3 + max3 + min3 + sma1 + energy3 + iqr3 + entropy3 + arCoeff12
#             # print(len(FEATURE))
#
#     #####################
#     # maxInds（）：幅度最大的频率分量的索引
#     # meanFreq（）：获得平均频率的频率分量的加权平均值
#     # skewness（）：频域信号的偏斜度
#     # f   # kurtosis（）：频域信号的峰度
#     # bandsEnergy（）：每个窗口的FFT的64个bin内的频率间隔的能量。
# #348
#     #####################79*3
#     for i in FreqfeatureList:
#         mean33,std33,mad33,max33,min33,sma11,energy33,iqr33,entropy33,maxinds33,meanFreq33,skewnessAndKurtosis66,bandsEnergy42=[],[],[],[],[],[],[],[],[],[],[],[],[]
#         if len(i)==3:
#             for j in range(len(i)):
#                 mean33.append(Meansig(i[j]))
#                 std33.append(std(i[j]))
#                 mad33.append(MAD(i[j]))
#                 min33.append(sigMIN(i[j]))
#                 max33.append(sigMAX(i[j]))
#                 energy33.append(energy(i[j]))
#                 iqr33.append(iqrs(i[j]))
#                 entropy33.append(sigEntropy(i[j]))
#                 maxinds33+=[maxinds(i[j],fs)]
#                 meanFreq33.append(meanFreq(i[j]))
#                 skewnessAndKurtosis66+=SkewAndKur(i[j])
#                 bandsEnergy42+=bandEnergy(i[j],len(i[j]))
#             sma11.append(sigSMA(i))
#             FEATURE += mean33 + std33 + mad33 + min33 + max33 + sma11 + energy33 + iqr33 + entropy33 + maxinds33 + meanFreq33 + skewnessAndKurtosis66 + bandsEnergy42
#             # print(len(FEATURE))
#         else:
#             mean33.append(Meansig(i))
#             std33.append(std(i))
#             mad33.append(MAD(i))
#             min33.append(sigMIN(i))
#             max33.append(sigMAX(i))
#             sma11.append(sigSMA(i))
#             energy33.append(energy(i))
#             iqr33.append(iqrs(i))
#             entropy33.append(sigEntropy(i))
#             maxinds33 += [maxinds(i, fs)]
#             meanFreq33.append(meanFreq(i))
#             skewnessAndKurtosis66 += SkewAndKur(i)
#             FEATURE+=mean33+std33+mad33+min33+max33+sma11+energy33+iqr33+entropy33+maxinds33+meanFreq33+skewnessAndKurtosis66
#
#     # angle（）：矢量之间的角度。
#     # 555
#     # angle(tBodyAccMean, gravity)
#     tBodyAccMean=calMEAN(tBodyAcc)
#     gravity=calMEAN(tGravity)
#     angle1=angle(tBodyAccMean,gravity)
#     FEATURE.append(angle1)
#     # 556
#     # angle(tBodyAccJerkMean), gravityMean)
#     tBodyAccJerkMean=calMEAN(tBodyAccJerk)
#     angle2=angle(tBodyAccJerkMean,gravity)
#     FEATURE.append(angle2)
#     # 557
#     # angle(tBodyGyroMean, gravityMean)
#     tBodyGyroMean=calMEAN(tgyro)
#     angle3=angle(tBodyGyroMean,gravity)
#     FEATURE.append(angle3)
#     # 558
#     # angle(tBodyGyroJerkMean, gravityMean)
#     tBodyGyroJerkMean=calMEAN(tBodyGyroJerk)
#     angle4=angle(tBodyGyroJerkMean,gravity)
#     FEATURE.append(angle4)
#     # 559
#     # angle(X, gravityMean)
#     angle5=angle('X',gravity)
#     FEATURE.append(angle5)
#     # 560
#     # angle(Y, gravityMean)
#     angle6 = angle('Y', gravity)
#     FEATURE.append(angle6)
#     # 561
#     # angle(Z, gravityMean)
#     angle7 = angle('Z', gravity)
#     FEATURE.append(angle7)
#     FEATURE=norm(FEATURE)
#     return FEATURE
# # df=pd.read_csv('trainACCandGYRO_100000.csv',header=None,sep=',')
# def generrate_signal(dir):
#     df=pd.read_csv(dir,header=None,sep=',')#dir='Gyro_t.csv'
#     L=len(df[2][1:])
#     accsig,gyrosig,gravitysig=[],[],[]
#     for i in range(2,5):
#         tem=[]
#         for j in range(1,len(df[i])):
#             tem.append(float(df[i][j]))
#             #+random.gauss(mu,sigma)
#         gravitysig.append(tem)
#     for i in range(5,8):
#         tem=[]
#         for j in range(1,len(df[i])):
#             tem.append(float(df[i][j]))
#             #+random.gauss(mu,sigma)
#         gyrosig.append(tem)
#     for i in range(8,11):
#         tem=[]
#         for j in range(1,len(df[i])):
#             tem.append(float(df[i][j]))
#             #+random.gauss(mu,sigma)
#         accsig.append(tem)
#     labels=df[13][1:]
#     x = np.linspace(0, 1, len(df[1][1:]))
#     return accsig,gyrosig,gravitysig,x,labels

# # print(df[2][1])
# # df=df[2:]
# # print(df[])
# # read_data=df[ ~df['type'].str.contains('gravity.z')]

# df = pd.read_csv('data415/Train418.csv', header=None, sep=',')
# print(df[3])
# df = pd.read_csv('data415/Train319.csv', header=None, sep=',')
# print(len(df[2]),'&&&&&&&&&&&&&&&&&&&&&&&&&&',df[3],'@@@@@@@@@@@@@@@@@@@',df[564])
# df = pd.read_csv('FeaturedTest.csv', header=None, sep=',')
# print(len(df[560]),df[0],'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',df[1])
# accsig,gyrosig,gravitysig,x=generrate_signal('GA20.csv')
# print(accsig[0][0],accsig[1][1],accsig[2][0],gyrosig[0][0],gyrosig[0][0],gyrosig[1][0],gravitysig[2][0],gravitysig[0][1],gravitysig[0][2],x)
# cwd = os.getcwd()
# read_path='HARdata'
# save_name='AllData222.csv'
# os.chdir(read_path)
# csv_name_list=os.listdir()
# df=pd.read_csv(csv_name_list[0])
# df=df[1:]
# df.to_csv('D:\pycharm\HAR_CNN\csv'+save_name,index=False)
# for i in range(1,360):
#     df = pd.read_csv(csv_name_list[i])
#     df=df[1:]
#     df.to_csv('D:\pycharm\HAR_CNN\csv' + save_name,index=False, header=False, mode='a+')

# print(accsig[0][0],accsig[1][1],accsig[2][0],gyrosig[0][0],gyrosig[0][0],gyrosig[1][0],gravitysig[2][0],gravitysig[0][1],gravitysig[0][2],x)
# accsig,gyrosig,gravitysig,x,labels=generrate_signal('ss.csv')
# fs=50
# windowsize=2*fs
# stride=fs
# accsig_idx,gyrosig_idx,gravitysig_idx=[[]]*3,[[]]*3,[[]]*3
# for i in range(0,200,fs):
#     for j in range(3):
#         accsig_idx[j]=accsig[j][i:i+windowsize]
#         gyrosig_idx[j]=gyrosig[j][i:i+windowsize]
#         gravitysig_idx[j]=gravitysig[j][i:i+windowsize]
#     label=labels[(i*2+windowsize)//2]
#     # feature=ALLFeature(fs,accsig_idx,gyrosig_idx,gravitysig_idx,x)
#     print(i,accsig_idx,gyrosig_idx,gravitysig_idx,label)
#

# df=pd.read_csv('csvAllData222.csv',header=None,sep=',')
# print(df[10])
# df=pd.read_csv('Testmydata.csv',header=None,sep=',')
# print(df[2],df[4])

# print(len(a))
# df=pd.read_table('csv/data_1600_gyro_phone.txt',sep=',',header = None)
# print(df)
# rows = [row for row in df]
# print(rows)
# import csv
# for i in range(1600,1651):
#     acctxtdir='csv/phone/accel'+'data_'+str(i)+'_accel_phone.txt'
#     gyrotxtdir='csv/phone/gyro'+'data_'+str(i)+'_gyro_phone.txt'
#
#     acc,gro=[],[]
#     with open(acctxtdir,'r') as csvfile:
#         reader =csv.reader(csvfile)
#         accrows = [row for row in reader]
#         for row in accrows:
#             for i in row[3:]:
#                 i=float(i)
#         acc.append(accrows)
#     with open(gyrotxtdir,'r') as csvfile:
#         reader =csv.reader(csvfile)
#         gyrorows = [row for row in reader]
#         for row in gyrorows:
#             for i in row[3:]:
#                 i=float(i)
#         gro.append(gyrorows)
#     labelDict={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':14,'P':15,'Q':16,'R':17,'S':18}
#     for j in range(0,min(len(acc),len(gro))):
#         for k in range(0,min(acc[j],gro[j])-250,100):
#             accsig,gyrosig=[[]]*3,[[]]*3
#             for h in range(k,k+200):
#                 accsig[0].append(acc[j][h][3])
#                 accsig[1].append(acc[j][h][4])
#                 accsig[2].append(acc[j][h][5])
#                 gyrosig[0].append(gro[j][h][3])
#                 gyrosig[1].append(gro[j][h][4])
#                 gyrosig[2].append(gro[j][h][5])
#                 label=labelDict[gro[j][100+h][1]]
#
#
# df=pd.read_csv('walk1111.csv',sep=',',header=None)
# print(df[0])
# # print(rows[0])
#
#



import csv
# list1=[1,1,1,1,1,1]
# list2=[2,2,2,2,2,2]
# data=[[3,3,3,3,3,3],
#       [4,4,4,4,4,4]]
# with open("test3.csv","a",newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=' ')
#     # writer.writerow([None,None,None,None,None,None])
#     writer=csv.writer(csvfile)
#     writer.writerow(list1)
#     writer.writerow(list2)
#
#     csvfile.close()
#
#
#
#
#

#
# df=pd.read_csv('mydataset/stand317_1.csv',header=None,sep=',')
#
# print(df[1
#       ][0])
#
#
#
#
#
"""
     standing 0;walk 1;laying 2;run 3；down 4;up 5
"""
def addlable():
    df = pd.read_csv('data422/CleanData/walk.csv', header=None, sep=',')
    # print(df[0],'************',df[1][0],df[1][1],df[1][2],'********************************',df[562])
    df['label']=1
    # print
    # df['label1'][0]='label'

    # for i in range(len(df[1])):
    #     df[562][i]=4
    df.to_csv('data422/CleanData/feature422.csv',mode='a',header=False)
    #
# addlable()
# addlable()
def spl():
    df = pd.read_csv('data422/CleanData/feature422.csv', header=None, sep=',')
    df1=df.sample(frac=1)
    Len=len(df1[1])
    point=int(Len*0.8)
    # df2=df.sample(frac=0.3)
    # # df=df[:100000]
    # # df=df.drop([0,1,2])
    # # df=df[:200000]
    train=df1.iloc[:point]
    test=df1.iloc[point:]
    train.to_csv('data422/CleanData/Train422.csv')
    test.to_csv('data422/CleanData/Test422.csv')
spl()
import threading as td
import time
import multiprocessing as mp
# def t1():
#     print('t1 start')
#     for i in range(10):
#         print('hahahah')
#         time.sleep(0.1)
#     print('t1 stop')
#
# def t2():
#     print('t2 start')
#     # time.sleep(0.1)
#     print('t2 stop')
# def multicore():
#     pool=mp.Pool(processes=2)
# def main():
#     p1=mp.Process(target=t1)
#     p2=mp.Process(target=t2)
#     p1.start()
#     # process.append(p1)
#     p2.start()
#     # process.append(p2)
#     # for i in process:
#     #     i.join()
#     # p1.start()
#     # p2.start()
#     p1.join()
#     p2.join()

# if __name__=='__main__':
    # dic = {
    #     'A': 1,
    #     'B': 2,
    #     'C': 3,
    #     'D': 4,
    #     'E': 5,
    #     'F': 6,
    #     'G': 7,
    #     'H': 8,
    #     'I': 9,
    #     'J': 10,
    #     'K': 11,
    #     'L': 12,
    #     'M': 13,
    #     'N': 14,
    #     'O': 15,
    #     'P': 16,
    #     'Q': 17,
    #     'R': 18,
    #     'S': 19,
    #     'T': 20,
    #     'U': 21,
    #     'V': 22,
    #     'W': 23,
    #     'X': 24,
    #     'Y': 25,
    #     'Z': 26
    # }
    # s = input().strip()
    # s = str(s)
    # res = 0
    # L = len(s)
    # for i in range(L):
    #     res += dic[s[i]] * (26 ** (L - i - 1))
    # print(res)
    # while True:
    #     main()
    # import multiprocessing as mp
    # from multiprocessing import Queue
    # q=Queue(50)
    # a=[1]*20+[2]*20+[3]*20
    # # array=
    # # print(array[44])
    #
    # for i in range(len(a)):
    #     if q.full()==False:
    #         q.put(a[i])
    #     else:
    #         lis=[]
    #         q.get()
    #         q.put(a[i])
    # lis=[]
    # for i in range(50):
    #     lis.append(q.get())
    # print(lis)
    #.acquire()锁住
    #.release()释放
    # value=mp.Value('d')
    # array=mp.Array('f',)##只能一维
    # process=[]

# def main():
#      addded_thread=td.Thread(target=tq(),name='t1')
#      adddd=td.Thread(target=t2(),name='t2')
#      # addded_thread.start()
#      # addded_thread.join()
#      # adddd.start()
#      #
#      # adddd.join()
#      threads=[]
#      threads.append(addded_thread)
#      threads.append(adddd)
#      for t in threads:
#          t.start()
#          print('okok')
#          print(td.active_count())
#          print(td.enumerate())
# if __name__=='__main__':
#     main()
#
#
#
#
#
#
#


