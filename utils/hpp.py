import numpy as np
import numba as nb
import math
from collections import Counter
import time
from scipy.optimize import minimize
# import time
# import os
# from global_config import GLOBAL_PATH

class HPP:
    def __init__(
        self,
        #events,
        cfg,
        content_num,
        #init_para_vector,
        if_print   = False
    ):
        #超参数
        self.PPF_delta           = cfg['PPF_delta']
        #self.slot_interval   = cfg['slot_interval']
        #self.update_interval = cfg['update_interval']
        #self.PAC_D           = cfg['PAC_D']

        #数据集相关参数
        self.all_t       = cfg['all_t'] #最大时间
        # self.t_o         = 0 #更新时间点
        #self.events     = events
        self.content_num = content_num
        # self.ED_index    = int(events['ed'].drop_duplicates().values)
        self.sumKernel  = np.zeros((self.content_num,self.all_t), dtype = np.float64) #核函数表
        self.sumI1      = np.zeros((self.content_num,self.all_t), dtype = np.float64) #部分核函数积分表
        self.eventNum_t = np.zeros((self.content_num,self.all_t), dtype = np.int32) #历史事件请求次数表

        #模型参数
        # self.beta      = np.zeros(self.content_num)
        # self.p         = np.zeros((self.content_num,self.PAC_D))
        # self.q         = np.zeros((self.content_num,self.PAC_D))
        # self.beta_zero = np.ones(self.content_num) * 1e-9
        # self.p_zero    = np.ones((self.content_num,self.PAC_D)) * 1e-9
        # self.q_zero    = np.ones((self.content_num,self.PAC_D)) * 1e-9


        #训练相关设置
        #self.ifLoadKernel = cfg['ifLoadKernel']
        #self.train_level  = cfg['train_level']
        #self.ifPrint      = if_print
        #self.iteration    = [0]

        #初始化核函数 
        #self.InitKernel(events) #改为并行初始化

    '''
    @zhangxz
    初始化kernel和积分项
    '''
    # def InitKernel(self,events): 
    #     self.sumKernel  = np.zeros((self.content_num,self.all_t), dtype = np.float64)
    #     self.sumI1      = np.zeros((self.content_num,self.all_t), dtype = np.float64)
    #     self.eventNum_t = np.zeros((self.content_num,self.all_t), dtype = np.int32)

    #     eventArray = events.to_numpy(dtype=np.int32)
    #     items = eventArray[:, 1]  # 获取所有事件的项目列表
    #     taus = eventArray[:, 3]   # 获取所有事件的tau列表
    #     np.add.at(self.eventNum_t, (items, taus), 1) # 使用numpy的bincount函数来一次性增加事件数量

        
    #     kernel_name = f"tmp_data/kernel_data/{self.train_level}-ed_{self.ED_index}-timeSlot_{self.slot_interval}-delta_{self.PPF_delta}-content_{self.content_num}.npy"
    #     integral_name = f"tmp_data/integral_data/{self.train_level}-ed_{self.ED_index}-timeSlot_{self.slot_interval}-delta_{self.PPF_delta}-content_{self.content_num}.npy"
    #     kernel_path = os.path.join(GLOBAL_PATH,kernel_name)
    #     integral_path = os.path.join(GLOBAL_PATH,integral_name)

    #     # 获取文件夹路径
    #     folder_path = os.path.dirname(kernel_path)
    #     # 如果文件夹路径不存在，则创建它
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     # 获取文件夹路径
    #     folder_path = os.path.dirname(integral_path)
    #     # 如果文件夹路径不存在，则创建它
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     if self.ifLoadKernel and os.path.exists(kernel_path) and os.path.exists(integral_path):
    #         self.sumKernel = np.load(kernel_path)
    #         self.sumI1 = np.load(integral_path)
    #     else:
    #         #增量计算核函数对应表
    #         for tau in range(1,self.all_t):
    #             self.sumKernel[:,tau] = (self.eventNum_t[:,tau-1] + self.sumKernel[:,tau-1]) * np.exp(-self.PPF_delta)
    #             new_event_integral = self.eventNum_t[:,tau-1] * ((1 - np.exp(- self.PPF_delta * self.update_interval)) / self.PPF_delta)
    #             self.sumI1[:,tau] = (new_event_integral + self.sumI1[:,tau-1]) * np.exp(-self.PPF_delta) 
    #         if not os.path.exists(kernel_path):
    #             np.save(kernel_path,self.sumKernel)
    #         if not os.path.exists(integral_path):
    #             np.save(integral_path,self.sumI1)

    def setKernel(self,kerenls):
        self.eventNum_t = kerenls[0]
        self.sumKernel  = kerenls[1]
        self.sumI1      = kerenls[2]
        

    # def set_t_o(self,t_o):
    #     self.t_o = t_o

    # def calSumI(self):
    #     #sumI1 = np.zeros(self.content_num, dtype = np.float32)
    #     #for tau in range (0, self.t_o-self.update_interval):
    #     #    sumI1 += self.eventNum_t[:,tau] * ( ( np.exp(self.PPF_delta * self.update_interval) - 1) * np.exp(-self.PPF_delta*(self.t_o - tau)) ) / self.PPF_delta
    #     sumI2 = np.zeros(self.content_num, dtype = np.float64)
    #     for tau in range (self.t_o-self.update_interval, self.t_o):
    #         sumI2 += self.eventNum_t[:,tau] * (1 - np.exp(-self.PPF_delta*(self.t_o - tau))) / self.PPF_delta
    #     return self.sumI1[:,self.t_o-self.update_interval] + sumI2
        #return sumI1 + sumI2

    # def likelihood(self, beta, p, q, events, disp = False):
    #     all_begin_time = time.time()

    #     #获取最新的参数
    #     omega = p @ q.T

    #     #onEventArray = self.events[(self.events['time'] >=self.t_o-self.update_interval) & (self.events['time'] < self.t_o)].copy().to_numpy(dtype=np.int32)
    #     onEventArray = events

    #     #计算强度函数
    #     def calIntensity(eventArray, beta, omega, p, q, sumKernel, content_num, PAC_D):
    #         content_indices = eventArray[:, 1]
    #         taus = eventArray[:, 3]
    #         # 计算强度函数
    #         Inten_tau = beta[content_indices] + np.sum(omega[content_indices, :] * sumKernel[:,taus].T, axis=1)
    #         sumIntensityTerm = np.sum(np.log(Inten_tau))
    #         betaGrad1 = np.zeros_like(beta)
    #         pGrad1 = np.zeros_like(p)
    #         qGrad1 = np.zeros_like(q)
    #         # 使用NumPy的矩阵操作计算梯度
    #         inv_Inten_tau = 1 / Inten_tau
    #         betaGrad1 = np.bincount(content_indices, weights=inv_Inten_tau, minlength=content_num)

    #         for d in range(PAC_D):
    #             tmp_pGrad1 = (q[:,d].T @ sumKernel[:,taus]).T / Inten_tau
    #             pGrad1[:,d] = np.bincount(content_indices, weights=tmp_pGrad1, minlength=content_num)
    #             qGrad1[:,d] = sumKernel[:,taus] @ (p[content_indices,d] / Inten_tau) 

    #         return sumIntensityTerm, betaGrad1, pGrad1, qGrad1

    #     sumIntensityTerm, betaGrad1, pGrad1, qGrad1  = \
    #         calIntensity(onEventArray, beta, omega, p, q, self.sumKernel, self.content_num, self.PAC_D)

    #     #积分项第一项
    #     sumBetaTerm = self.update_interval * np.sum(beta) 

    #     #积分项第二项
    #     sumI_to = self.calSumI()
    #     sumOmegaTerm = np.sum( omega @ sumI_to )

    #     #总积分项
    #     sumIntegralTerm = sumBetaTerm + sumOmegaTerm

    #     #总目标
    #     obj = sumIntensityTerm - sumIntegralTerm
        
    #     if self.ifPrint:
    #         print(self.iteration,'Iteration likelihood: ', end = "")
    #         print('{',round(obj),'}=', \
    #               -round(sumIntensityTerm,4),'+',round(sumIntegralTerm,4) )
    #         print(self.iteration,'Iteration likelihood finish use time:', round(time.time() - all_begin_time,2), 's')

    #     return - obj , list([betaGrad1, pGrad1, qGrad1, sumI_to])

    # def gradient(self, beta, p, q, grad_tmp, disp=False):

    #     all_begin_time = time.time()
        
    #     betaGrad = np.zeros_like(beta)
    #     pGrad    = np.zeros_like(p)
    #     qGrad    = np.zeros_like(q)

    #     betaGrad1 = grad_tmp[0]
    #     pGrad1    = grad_tmp[1]
    #     qGrad1    = grad_tmp[2]
    #     sumI_to   = grad_tmp[3]

    #     #计算 beta 偏导数
    #     betaGrad = betaGrad1 - self.update_interval

    #     #计算 p q 偏导数
    #     expanded_sumI_to = np.repeat(sumI_to[:, np.newaxis], self.PAC_D, axis=1)
    #     pGrad2 = np.sum(q * expanded_sumI_to, axis=0)
    #     qGrad2 = np.zeros_like(q)
    #     sum_p = np.sum(p,axis=0)
    #     for d in range(self.PAC_D):
    #         pGrad[:,d] = pGrad1[:,d] - pGrad2[d]
    #         qGrad2[:,d] = sum_p[d] * sumI_to
    #     qGrad = qGrad1 - qGrad2

    #     #计算 p q 偏导数
    #     #expanded_sumI_to = np.tile(self.sumI_to[:,np.newaxis],(1,self.PAC_D))
    #     #sum_q  = np.sum(q * expanded_sumI_to, axis=0)
    #     #pGrad2 = np.tile(sum_q,(self.pGrad1.shape[0],1))
    #     #pGrad  = self.pGrad1- pGrad2
    #     #sum_p  = np.tile(np.sum(p,axis=0),(self.qGrad1.shape[0],1))
    #     #qGrad2 = expanded_sumI_to * sum_p
    #     #qGrad  = self.qGrad1 - qGrad2


    #     if self.ifPrint:       
    #         print(self.iteration,'Iteration Gradient use time:', round(time.time() - all_begin_time,2), 's')

    #     return -betaGrad, -pGrad, -qGrad

    # def get_likelihood_gradient(self, beta, p, q, events, disp=False):
    #     loss, grad_tmp = self.likelihood(beta, p, q, events)
    #     neg_betaGrad, neg_pGrad, neg_qGrad = self.gradient(beta, p, q, grad_tmp)
    #     return loss, neg_betaGrad, neg_pGrad, neg_qGrad


class HRS:
    def __init__(
        self,
        ed_index,
        events,
        cfg,
        content_num,
        #init_para_vector,
        if_print   = False
    ):
        #超参数
        self.delta     = 0.5
        self.delta1    = 1.5
        self.threshold = 12  #创造负激励的阈值，经过阈值个timewindow 作为一个负激励事件
        self.scale     = 1e-9
        self.iteration    = [ed_index,0,0]
        # self.slot_interval   = cfg['slot_interval']
        #self.update_interval = cfg['update_interval']
        self.if_print = if_print

        #数据集相关参数
        self.allT       = cfg['all_t'] #最大时间
        self.numAllItem = content_num
        self.ED_index    = ed_index
        # self.sumK0  = np.zeros((self.numAllItem,self.allT), dtype = np.float64) #核函数表
        # self.sumK1  = np.zeros((self.numAllItem,self.allT), dtype = np.float64) #核函数表
        taEvents = events[events['day'] < cfg['test_day']] 
        taEventsArray = taEvents[['ed', 'i', 'time']].to_numpy(dtype=np.int32)
        tsEvents = events[(events['day'] >=cfg['test_day']) & (events['day'] < self.allT)]
        tsEventsArray = tsEvents[['ed', 'i', 'time']].to_numpy(dtype=np.int32)  
        # self.sumI1      = np.zeros((self.numAllItem,self.all_t), dtype = np.float64) #部分核函数积分表
        # self.eventNum_t = np.zeros((self.numAllItem,self.all_t), dtype = np.int32) #历史事件请求次数表

        #模型参数
        self.b             = np.ones(self.numAllItem)
        self.gamma         = np.ones(self.numAllItem)
        self.alpha         = np.ones(self.numAllItem)
        self.omega         = np.ones(self.numAllItem)
        self.b_zero        = np.ones(self.numAllItem) * 1e-9
        self.gamma_zero    = np.ones(self.numAllItem) * 1e-9
        self.alpha_zero    = np.ones(self.numAllItem) * 1e-9
        self.omega_zero    = np.ones(self.numAllItem) * 1e-9
        self.penalty_b     = 1e5
        self.penalty_gamma = 1e5
        self.penalty_omega = 1e5
        self.penalty_alpha = 1e5

        self.InitNe(taEventsArray,tsEventsArray) #初始化 Ne Nd 项 所有数据都需要
        self.InitKernel()

    def x_to_para(self, x):
        begin = 0 
        end = begin + np.cumprod(self.b.shape)[-1]
        self.b = x[begin : end].reshape(self.b.shape)
        self.b_zero [self.b <= 0] = self.b_zero[self.b <= 0] / 10
        self.b[self.b <= 0] = self.b_zero[self.b <= 0]
        begin = end 
        
        end = begin + np.cumprod(self.alpha.shape)[-1]
        self.alpha = x[begin : end].reshape(self.alpha.shape)
        self.alpha_zero[self.alpha <= 0] = self.alpha_zero[self.alpha <= 0] / 10
        self.alpha[self.alpha <= 0] = self.alpha_zero[self.alpha <= 0]        
        begin = end
        
        end = begin + np.cumprod(self.omega.shape)[-1]
        self.omega = x[begin : end].reshape(self.omega.shape)
        self.omega_zero[self.omega <= 0] = self.omega_zero[self.omega <= 0] / 10
        self.omega[self.omega <= 0] = self.omega_zero[self.omega <= 0]
        begin = end

        end = begin + np.cumprod(self.gamma.shape)[-1]
        self.gamma = x[begin : end].reshape(self.gamma.shape)
        self.gamma_zero[self.gamma <= 0] = self.gamma_zero[self.gamma <= 0] / 10
        self.gamma[self.gamma <= 0] = self.gamma_zero[self.gamma <= 0]
        begin = end
    
    def para_to_x(self):
        return np.concatenate([self.b.ravel(), self.alpha.ravel(), self.omega.ravel(),self.gamma.ravel()])

    '''
    @zhangxz
    动态更新每条记录的强度函数时中间变量
    '''
    def InitNe(self,taEventsArray,tsEventsArray): 
        self.Ne             = np.zeros((self.numAllItem,self.allT), dtype = np.float32)
        self.timeNumE       = np.zeros((self.numAllItem,self.allT), dtype = np.float32)
        self.Nu             = np.zeros((self.numAllItem,self.allT), dtype = np.float32)
        self.timeNoE        = np.zeros((self.numAllItem,), dtype = np.float32)
        self.itemUploadTime = np.full((self.numAllItem), -1 ,dtype = np.int32)

        @nb.njit()
        def UpdateNumE(itemUploadTime,events,timeNumE):
            for i in range(events.shape[0]):
                item = events[i,1]
                time = events[i,2]
                timeNumE[item,time] += 1
                if itemUploadTime[item] < 0:
                    itemUploadTime[item] = int(time)

        UpdateNumE(self.itemUploadTime,taEventsArray,self.timeNumE)
        UpdateNumE(self.itemUploadTime,tsEventsArray,self.timeNumE)

        
        @nb.njit()
        def UpdateNeNu(allT,timeNumE,timeNoE,numAllItem,Ne,Nu,threshold):
            for t in range (1,allT):
                Ne[:,t] = timeNumE[:,t-1] + Ne[:,t-1]
                for item in range(numAllItem):
                    if timeNumE[item,t-1] == 0:
                        timeNoE[item] += 1
                        if timeNoE[item] >= threshold:
                            #timeNoE[item]=0
                            Nu[item,t] = 1
                    else:
                        timeNoE[item]=0

        UpdateNeNu(self.allT,self.timeNumE,self.timeNoE,self.numAllItem,self.Ne,self.Nu,self.threshold)
        
        self.neScale = math.ceil(np.max(self.Ne))
        self.Ne = self.Ne / self.neScale

    def InitKernel(self):
        #self.kernel = np.zeros(self.allT)
        self.sumMuKernel = np.zeros((self.numAllItem,self.allT)) 
        self.sumKernel   = np.zeros((self.numAllItem,self.allT)) # 存储每一条记录中 UI 元组多对应的强度函数中 g 函数项求和值 @zhangxz
        self.sumNeKernel = np.zeros((self.numAllItem,self.allT)) # 计算self.E_u_i中间变量 @zhangxz
        @nb.njit()
        def calKernel(numAllItem,allT,delta,delta1,Ne,Nu,itemUploadTime,timeNumE,alpha,sumKernel,sumMuKernel,sumNeKernel):
            kernel  = np.exp(-delta )
            kernel1 = np.exp(-delta1)
            for item in range(numAllItem):
                for t in range(1,allT):
                    sum_tmp = Nu[item,t-1] * kernel1
                    sumMuKernel[item,t] = sumMuKernel[item,t-1] * kernel1 + sum_tmp
                    sum_tmp = timeNumE[item,t -1 ] * kernel * np.exp(-alpha[item] * Ne[item,t -1 ]) 
                    sumKernel[item,t] = sum_tmp + sumKernel[item,t-1] * kernel
                    sumNeKernel[item,t] = sum_tmp * Ne[item,t-1 ] + sumNeKernel[item,t-1] * kernel

        calKernel(self.numAllItem,self.allT,self.delta,self.delta1,self.Ne,self.Nu,self.itemUploadTime,self.timeNumE,self.alpha,self.sumKernel,self.sumMuKernel,self.sumNeKernel)

    def OnlineUpdateSum(self):
        
        self.sumKernel  [:,self.onlineBeginT-self.tsLowT:self.onlineEndT] = 0
        self.sumNeKernel[:,self.onlineBeginT-self.tsLowT:self.onlineEndT] = 0
        '''
        @zhangxz
        迭代计算核函数求和项并存表
        '''
        @nb.njit()
        def calsumKernel(numAllItem,delta,beginTime,endTime,\
            timeNumE,Ne,alpha,sumKernel,sumNeKernel):
            kernel = np.exp(-delta)
            #sum_tmp = np.zeros(numAllItem)
            for item in range(numAllItem):
                for t in range(beginTime, endTime ):
                    sum_tmp = timeNumE[item,t - 1] * kernel * np.exp(-alpha[item] * Ne[item,t -1]) 
                    sumKernel  [item,t] = sum_tmp + sumKernel[item,t-1] * kernel
                    sumNeKernel[item,t] = sum_tmp * Ne[item,t-1 ] + sumNeKernel[item,t-1] * kernel
                        
        calsumKernel(self.numAllItem,self.delta,self.onlineBeginT+1,self.onlineEndT,\
            self.timeNumE,self.Ne,self.alpha,self.sumKernel,self.sumNeKernel)
    
    def onlineLoss(self, x, disp = False):
        all_begin_time = time.time()

        self.x_to_para(x)
        self.OnlineUpdateSum()

        self.Inten       = np.zeros((self.numAllItem,self.tsPeriodT))
        self.scaledInten = np.zeros((self.numAllItem,self.tsPeriodT))
        
        #计算强度函数和变化过后的强度函数
        @nb.njit()
        def calIntensity(
            beginTime,endTime,numAllItem,scale,
            b,omega,gamma,
            sumKernel,sumMuKernel,
            Inten,scaledInten
        ):
            for day_t in range(beginTime,endTime+1):
                for item in range(numAllItem):
                    I = b[item] + omega[item] * sumKernel[item,day_t] - gamma[item]*sumMuKernel[item,day_t]
                    Inten[item,day_t-beginTime] = I
                    scaledInten[item,day_t-beginTime] = scale * np.log(1 + 1e-15 + np.exp( I / scale)) if I < scale * 100 else I
        calIntensity(
            self.onlineBeginT,self.onlineEndT,self.numAllItem,self.scale,\
            self.b,self.omega,self.gamma,\
            self.sumKernel,self.sumMuKernel,\
            self.Inten,self.scaledInten)
        
        #L函数第一项
        sumIntensityTerm = np.sum(self.timeNumE[:,self.onlineBeginT:self.onlineEndT] * np.log(self.scaledInten))
        
        #L函数第二项，利用蒙特卡洛采样估计整体积分项
        systemIntensity = np.sum(self.scaledInten,axis = 0)


        @nb.njit()
        def calIntegral(sampleDay,systemIntensity):
            sumIntegralTerm = 0
            for day_n in sampleDay:
                sumIntegralTerm += systemIntensity[day_n]
            return sumIntegralTerm
        sumIntegralTerm =  calIntegral(self.sampleOLT,systemIntensity) * self.tsPeriodT  / self.onlineSampleTimes

        tmp_obj = -sumIntensityTerm  + sumIntegralTerm
        
        obj = tmp_obj \
            + self.penalty_b     * np.sum(self.b     * self.b    )\
            + self.penalty_alpha * np.sum(self.alpha * self.alpha)\
            + self.penalty_omega * np.sum(self.omega * self.omega)\
            + self.penalty_gamma * np.sum(self.gamma * self.gamma)
        
        if self.if_print:
            print(self.iteration,'Iteration likelihood: ', end = "")
            print('{',round(obj),'}=', \
                  -round(sumIntensityTerm,4),'+', round(sumIntegralTerm,4), \
                  ' + ', self.penalty_b,     '*', round(np.sum(self.b     * self.b    ),4), \
                  ' + ', self.penalty_alpha, '*', round(np.sum(self.alpha * self.alpha),4), \
                  ' + ', self.penalty_omega, '*', round(np.sum(self.omega * self.omega),4), \
                  ' + ', self.penalty_gamma, '*', round(np.sum(self.gamma * self.gamma),4),
                  )
            print(self.iteration,'Iteration likelihood finish use time:', round(time.time() - all_begin_time,2), 's')
        
        return obj

    def onlineGradient(self, x, disp=False):
        all_begin_time = time.time()

        bGrad     = np.zeros(self.numAllItem)
        alphaGrad = np.zeros(self.numAllItem)
        omegaGrad = np.zeros(self.numAllItem)
        gammaGrad = np.zeros(self.numAllItem)
        
        I = 1 / self.scaledInten
        
        @nb.njit()
        def calGrad(timeNumE,Nu,sampleTNum,sampleTimes,tsPeriodT,I,Inten,numAllItem,\
           beginTaT,scale,sumKernel,sumNeKernel,sumMuKernel,omega,\
                bGrad,alphaGrad,omegaGrad,gammaGrad):
            bIntenGrad     = np.zeros((numAllItem,tsPeriodT))
            alphaIntenGrad = np.zeros((numAllItem,tsPeriodT))
            omegaIntenGrad = np.zeros((numAllItem,tsPeriodT))
            gammaIntenGrad = np.zeros((numAllItem,tsPeriodT))
            for t in range(tsPeriodT):
                for item in range(numAllItem):
                    bIntenGrad[item] = np.exp(Inten[item,t] / scale) / ( 1 + np.exp(Inten[item,t] / scale) ) if Inten[item,t]< scale * 100 else 1
                    alphaIntenGrad[item,t] = - bIntenGrad[item,t] * omega[item] * sumNeKernel[item,t+beginTaT]
                    omegaIntenGrad[item,t] =   bIntenGrad[item,t] * sumKernel[item,t+beginTaT]
                    gammaIntenGrad[item,t] = - bIntenGrad[item,t] * sumMuKernel[item,t+beginTaT]

                    #bGrad[item]      += (timeNumE[item,t] + Nu[item,t])* I[item,t] * bIntenGrad[item,t]
                    bGrad[item]      += timeNumE[item,t+beginTaT]* I[item,t] * bIntenGrad[item,t]
                    bGrad[item]      -= (tsPeriodT * sampleTNum[t]  * bIntenGrad[item,t])/sampleTimes

                    #alphaGrad[item]  += (timeNumE[item,t] +Nu[item,t]) * I[item,t] * alphaIntenGrad[item,t]
                    alphaGrad[item]  += timeNumE[item,t+beginTaT]  * I[item,t] * alphaIntenGrad[item,t]
                    alphaGrad[item]  -= (tsPeriodT * sampleTNum[t]  * alphaIntenGrad[item,t])/sampleTimes

                    #omegaGrad[item]  += (timeNumE[item,t] +Nu[item,t]) * I[item,t] * omegaIntenGrad[item,t]
                    omegaGrad[item]  += timeNumE[item,t+beginTaT]  * I[item,t] * omegaIntenGrad[item,t]
                    omegaGrad[item]  -= (tsPeriodT * sampleTNum[t]  * omegaIntenGrad[item,t])/sampleTimes

                    #gammaGrad[item]  += (timeNumE[item,t] + Nu[item,t])  * I[item,t] * gammaIntenGrad[item,t]
                    gammaGrad[item]  += timeNumE[item,t+beginTaT]  * I[item,t] * gammaIntenGrad[item,t]
                    gammaGrad[item]  -= (tsPeriodT * sampleTNum[t]  * gammaIntenGrad[item,t])/sampleTimes
        
        calGrad(self.timeNumE, self.Nu, self.sampleOLTNum,self.onlineSampleTimes,self.tsPeriodT, I, self.Inten, self.numAllItem,\
            self.onlineBeginT,self.scale,self.sumKernel,self.sumNeKernel,self.sumMuKernel,self.omega,\
                bGrad,alphaGrad,omegaGrad,gammaGrad)

        #计算惩罚项
        bGrad     -= self.penalty_b     * 2 *self.b
        alphaGrad -= self.penalty_alpha * 2 *self.alpha
        omegaGrad -= self.penalty_omega * 2 *self.omega
        gammaGrad -= self.penalty_gamma * 2 *self.gamma

        all_grad = np.concatenate([bGrad.ravel(), alphaGrad.ravel(), omegaGrad.ravel(), gammaGrad.ravel()])
           
        if self.if_print:       
            print(self.iteration,'Iteration Gradient use time:', round(time.time() - all_begin_time,2), 's')

        return -1 * all_grad

    def onlineFit(self,disp = False, maxiter = 10, threshold = 9,period = 48, onlineBeginT = 288): #单次在线训练
        self.xBackup = self.para_to_x()
        # self.iterBackup = self.iteration[0]

        self.tsLowT = int(threshold / self.delta) 
        self.tsPeriodT = period #更新周期

        self.onlineBeginT = onlineBeginT
        self.onlineEndT = self.onlineBeginT + self.tsPeriodT

        #计算在线过程中每个时间的采样点数
        self.onlineSampleTimes = int(6000 * self.tsPeriodT) #HRS: 6000每小时
        self.sampleOLT = np.random.randint(self.tsPeriodT, size=self.onlineSampleTimes)
        sampleTNumDict = dict(Counter(self.sampleOLT))
        self.sampleOLTNum = np.zeros(self.tsPeriodT,dtype=int)
        for t in range(self.tsPeriodT):
            self.sampleOLTNum[t] = sampleTNumDict[t]

        
        x0 =  self.para_to_x()
        
        bnd_b     = np.cumprod(self.b.shape)    [-1]
        bnd_alpha = np.cumprod(self.alpha.shape)[-1] + bnd_b
        bnd_omega = np.cumprod(self.omega.shape)[-1] + bnd_alpha
        
        bnds      = [(0,None) for i in range(bnd_b)] + \
                    [(0,None) for i in range(bnd_b , bnd_alpha)] + \
                    [(0,None) for i in range(bnd_alpha , bnd_omega)] + \
                    [(0,None) for i in range(bnd_omega , len(x0))]
                            # 均无约束条件
        
        self.iters = 0
        def call_back(x):
            if self.if_print:
                #print(self.iters)
                #print(self.iteration,'Iteration callback, this iteration use time:', time.time() - self.last_time, 's')
                #print(self.iteration,'Iteration callback, totally use time:', time.time() - self.first_time, 's\n') 
                pass
            self.iteration[2] = self.iteration[2] + 1
            self.last_time = time.time()
            
        self.first_time = time.time()
        self.last_time = time.time()
        res = minimize( self.onlineLoss,
                        self.para_to_x(),
                        method = 'L-BFGS-B',
                        jac = self.onlineGradient,
                        bounds = bnds,
                        options = { 'ftol': 1e-6, 'maxiter': maxiter},
                        callback = call_back)
        self.x_to_para(res.x)

        if self.if_print:
            print(f'OnlineLoss end with iteration {self.iteration[0]}. OnlineLoss {self.onlineLoss(self.para_to_x())}.')
        self.iteration[2] = 0
        self.iteration[1] += 1
        return res