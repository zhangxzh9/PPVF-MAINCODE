import numpy as np
from utils.hpp import HPP, HRS
import random
from scipy.stats import pearsonr
from collections import OrderedDict
import copy

class ED(object):
    def __init__(self, index, cfg, content_num, events):   

        # 公共配置
        self.content_num = int(content_num) #视频总数
        self.ed_index = index

        self.predict_policy = cfg["predict_policy"] # 效用的预测模型（non, PPVF, GREED, INFOCOM：用的是HPP_PAC输出的强度函数，lfu:历史请求次数比例，opt:未来请求比例，...）
        self.fetch_policy   = cfg['fetch_policy'] # 预请求策略
        self.if_noise       = cfg['if_noise'] #是否加噪

        self.c_e        = cfg["c_e"]
        self.cache_list = [OrderedDict() for _ in range(cfg["hyper_paras_num"])] #有序缓存字典
        # self.epsilon    = cfg["epsilon"]#单轮请求决策消耗的隐私预算
        # self.xi         = cfg["xi"] #总的隐私预算
        self.remained_b = np.zeros((len(self.c_e),self.content_num),dtype=np.float32) # 目前隐私预算消耗的比例
        # self.remained_action_times = np.full(fill_value=int(cfg['xi']/),shape=(len(self.c_e),self.content_num),dtype=np.int32) # 目前隐私预算消耗的比例

        self.n          = 1 # 请求的次数
        self.actions    = np.zeros((len(self.c_e),self.content_num),dtype=np.int64) 

        if self.if_noise:
            self.sum_XY = np.zeros((self.content_num, self.content_num))  
            self.sum_X = np.zeros(shape=self.content_num)
            self.sum_X_square = np.zeros(shape=self.content_num)
            # self.sum_X = np.zeros((self.content_num, 1))  
            # self.sum_X_square = np.zeros((self.content_num, 1)) 

        if self.fetch_policy in ["PPVF","INFOCOM","GREED","RANDOM"]:# 缩放函数所需
            self.Up_bound  = 1 # 效用上限
            self.Low_bound = 0.1 # 效用下限
            self.B0        = 1 / (1 + np.log(self.Up_bound / self.Low_bound)) #最低阈值
        elif self.fetch_policy in ["OPT","LRU","FIFO","LFU"]:
            pass
        else:
            raise ValueError(f"not this fetch policy: {self.fetch_policy}")
        
        self.density = np.ones(self.content_num) / self.content_num
        if self.predict_policy in ["HPP_PAC","HPP_SELF"]:
            self.HPP = HPP(content_num = self.content_num, cfg = cfg, if_print = False)
        elif self.predict_policy in ["HRS"]:
            self.HPP = HRS(ed_index = self.ed_index, events = events, content_num = self.content_num, cfg = cfg)
        elif self.predict_policy in ["HIS"]:
            self.content_count = np.ones(shape=self.content_num,dtype=np.int32)
        elif self.predict_policy in ["NO_MODEL","FUT","HIS_ONE"]:
            pass
        else:
            raise ValueError(f"not this predict policy: {self.predict_policy}")

    def get_hitrate(self, ed_slot_trace):
        def if_success(x,cachingSet):
            if x['i'] in cachingSet:
                return 1
            else:
                return 0
        ed_click = ed_slot_trace.shape[0]
        #统计命中率
        if ed_click != 0:
            ed_hit = []
            if self.fetch_policy in ["GREED","INFOCOM","PPVF","RANDOM"]:
                for cache in self.cache_list:
                    ed_hit.append(int(np.sum(ed_slot_trace.apply(if_success, axis = 1, cachingSet=set(cache.keys())))))
            elif self.fetch_policy in ["LRU","LFU","FIFO"]:
                for cache_index in range(len(self.cache_list)):
                    ed_hit.append(int(np.sum(ed_slot_trace.apply(if_success, axis = 1, cachingSet=set(self.cache_list[cache_index].keys())))))
                    self.update_cache(cache_index=cache_index,redundant_request=None,ed_slot_trace=ed_slot_trace)
                    # ed_hit.append(int(self.update_cache(cache_index=cache_index,redundant_request=None,ed_slot_trace=ed_slot_trace)))
            else:
                raise ValueError(f"not this fetch policy: {self.fetch_policy}")
            ed_hit = np.array(ed_hit,dtype=np.int64)
        else:
            ed_hit = np.zeros(len(self.cache_list),dtype=np.int64)
        return ed_click, ed_hit


    def update_density(self, density):
        self.density = density

    def calculate_pearson_correlation(self):
        #data         = data[:,np.newaxis]
        self.n            += 1
        self.sum_XY       += self.density @ self.density.T
        self.sum_X        += self.density
        self.sum_X_square += self.density * self.density
        if self.n > 2:
            assert (self.n*self.sum_X_square>=self.sum_X * self.sum_X).all(), [self.n,self.ed_index]
            sigma_X = np.sqrt(self.n * self.sum_X_square - (self.sum_X * self.sum_X))
            denominator =  sigma_X @ sigma_X.T
            numerator = self.n * self.sum_XY - (self.sum_X @ self.sum_X.T) 
            psi = numerator / denominator
        else:
            psi = np.eye(self.content_num,dtype=np.float64)
        return psi

    def get_redundant_request(self, cache_index, f_e_total, epsilon, xi, psi): 
        #归一化
        def scale_value(value, input_min, input_max, output_min, output_max):
            # 输入区间范围
            input_range = input_max - input_min
            if input_range == 0:
                return value
            # 输出区间范围
            output_range = output_max - output_min
            # 计算映射后的值
            scaled_value = ((value - input_min) / input_range) * output_range + output_min
            return scaled_value

        def get_th(remained_b): # 计算阈值（基于阈值的在线算法相关）
            if remained_b < self.B0:
                return self.Low_bound
            else:
                return (((self.Up_bound * np.e) / self.Low_bound)**remained_b) * (self.Low_bound / np.e)

        np.random.seed(self.ed_index)
        # np.random.seed(0)
        scaled_density = scale_value(self.density, self.density.min(), self.density.max(), self.Low_bound, self.Up_bound) 

        cache = self.cache_list[cache_index]
        action = np.zeros(shape=self.content_num,dtype=np.int8) #当前slot的缓存action
        f_k = 0 
        if self.fetch_policy == "INFOCOM": #infocom2022 + HPP_PAC
            sampled_indexes = np.random.choice(range(self.content_num), size = self.content_num, replace = False) # 无放回随机采样
            for i in sampled_indexes:
                if f_k + 1 > f_e_total:
                    break
                # elif i in cache: #在缓存中就不请求
                #     pass
                elif (self.remained_b[cache_index,i] / xi) <= self.B0: #低于阈值，必然请求
                    f_k += 1
                    action[i] = 1
                    self.remained_b[cache_index,i] = self.remained_b[cache_index,i] + epsilon  
                elif (scaled_density[i] / epsilon > get_th(self.remained_b[cache_index,i] / xi )) and epsilon <= xi - self.remained_b[cache_index,i]:
                    f_k += 1
                    action[i] = 1
                    self.remained_b[cache_index,i] = self.remained_b[cache_index,i] + epsilon  
                else:
                    action[i] = 0
        else:
            raise ValueError(f"not such fetch policy: {self.fetch_policy} or {self.fetch_policy} not privacy budget requirement")

        if self.if_noise:
            action_indices  = np.argwhere(action==1)[:,0] #获取有隐私预算分配的视频index
            redundant_action = np.zeros(shape=self.content_num,dtype=np.int8) # 冗余请求决策

            psi[np.abs(psi) < 0.95] = 0 #根据阈值修改相关性

            if len(action_indices)>0:
                action_density  = self.density[action_indices]
                #计算每个视频的敏感度
                sensitivity = psi[action_indices,action_indices] @ action_density.T
                #全局敏感度
                sen_c = sensitivity.max()
                # 计算指数机制的概率
                probabilities   = np.exp(epsilon * action_density / (2 * sen_c))
                probabilities  /= np.sum(probabilities)  # 归一化到概率分布
                random_videos_indices = np.random.choice(action_indices, size=len(action_indices), p=probabilities, replace=True)
                redundant_action[random_videos_indices] = 1
        else:
            redundant_action = action
            # updated_noise_paras = None
            self.remained_b = np.zeros((len(self.c_e),self.content_num),dtype=np.float32)
        return redundant_action

    def update_cache(self, cache_index, redundant_request, ed_slot_trace):
        action = np.zeros(shape=self.content_num,dtype=np.int8) #当前slot的缓存action
        c_k = 0
        hits = 0
        cache = self.cache_list[cache_index]

        if self.fetch_policy in ["INFOCOM"]:
            action[list(set(ed_slot_trace['i']))] = 1
            redundant_request_indices = np.argwhere(redundant_request==1)[:,0]
            action[redundant_request_indices] = 1
            for video in cache.keys():
                action[video] = 0
                cache[video] = self.density[video]
            request_indices = np.argwhere(action==1)[:,0]
            for video,den_i in zip(request_indices,self.density[request_indices]):
                if len(cache) >= self.c_e[cache_index]:
                    # 找到强度最低的视频并删除
                    min_index = min(cache, key=cache.get)
                    if den_i>cache[min_index]:
                        del cache[min_index]
                        # 插入新视频或更新已存在视频的强度值
                        cache[video] = den_i
                else:
                    cache[video] = den_i
        else:
            raise ValueError(f"not such fetch policy: {self.fetch_policy}")
        self.actions[cache_index] += action
        return hits

    def get_noise_paras(self):
        if self.fetch_policy in ["INFOCOM"]:
            return (self.n, self.sum_XY, self.sum_X,self.sum_X_square)
        else:
            raise ValueError("not this fetch_policy {}.".format({self.fetch_policy}))

    def update_b(self, cache_index, remained_b):
        self.remained_b[cache_index,:] = remained_b

    def update_noise_paras(self, updated_buffer):
        if self.if_noise == True:
            if  self.fetch_policy in ["INFOCOM"]:
                self.n = updated_buffer[0]
                self.sum_XY = updated_buffer[1]
                self.sum_X = updated_buffer[2]
                self.sum_X_square = updated_buffer[3]
            else:
                raise ValueError(f"not this fetch_policy {self.fetch_policy} or {self.fetch_policy} can not add noise")
        else:
            raise ValueError("No noise is added, so no need to update the noise parameters")


