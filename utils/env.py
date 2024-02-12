import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Lock
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import os
import time

from global_config import GLOBAL_PATH, DATA_PATH
from utils.edge import ED
from utils.para_server import PS
from utils.parallel_cal_functions import init_ed_kernel, get_redundant_request, HRS_online_traning


class Env(object):
    def __init__(self, trace, cfg):
        # PS和ED初始化
        self.index2eds      = {} # ed字典
        self.ps             = PS(trace = trace, cfg=cfg, index2eds=self.index2eds) #在线训练聚合参数服务器
        self.maxProgressNum = cfg.maxProgressNum #最大并行进程数限制
        self.run_way        = cfg.run_way
        self.fetch_policy   = cfg.fetch_policy
        self.predict_policy = cfg.predict_policy
        self.if_noise       = cfg.if_noise
        self.maxiter        = cfg.maxiter
        
        # 数据集参数
        self.content_num     = trace.content_num #视频数量
        self.all_day         = cfg.all_day #总天数
        self.test_day        = cfg.test_day #开始测试的日期（天）
        self.update_interval = cfg.update_interval #HPP在线更新间隔（slot）
        self.all_t           = cfg.all_t # slot的总数
        self.test_begin_t    = self.test_day * cfg.day_2_time  #开始在线训练和测试的slot
        # self.t_o             = 0 #更新时间点

        #在线算法参数
        # self.Up_bound  = 1 # 效用上限
        # self.Low_bound = 0.1 # 效用下限
        # self.B0        = 1 / (1 + np.log( self.Up_bound / self.Low_bound)) #最低阈值

        self.test_hyper_paras = cfg.test_hyper_paras
        self.f_e              = cfg.f_e
        self.epsilon          = cfg.epsilon
        self.xi               = cfg.xi

        if self.test_hyper_paras =="f_e":
            self.hyper_paras_num = len(cfg.f_e)
            self.c_e = [cfg.c_e_ratio * self.content_num for _ in range(self.hyper_paras_num)]
        elif self.test_hyper_paras =="c_e":
            self.hyper_paras_num = len(cfg.c_e_ratio)
            self.c_e = [int(ratio * self.content_num) for ratio in cfg.c_e_ratio]
        elif self.test_hyper_paras =="epsilon":
            self.hyper_paras_num = len(cfg.epsilon)
            self.c_e = [cfg.c_e_ratio * self.content_num for _ in range(self.hyper_paras_num)]
        elif self.test_hyper_paras =="xi":
            self.hyper_paras_num = len(cfg.xi)
            self.c_e = [cfg.c_e_ratio * self.content_num for _ in range(self.hyper_paras_num)]
        else:
            raise ValueError(f"not such {self.test_hyper_paras}")

        #其他配置
        self.trace        = trace
        self.model_path   = cfg.model_path
        self.model_config = f"{cfg.predict_policy}-{cfg.train_level}-all_t:{self.all_t}-update_interval:{self.update_interval}-PPF_delta:{cfg.PPF_delta}-maxiter:{cfg.maxiter}-content_num:{self.content_num}/"

        # tensorborad
        self.summary_writer  = cfg.summary_writer
        self.actions_path    = cfg.actions_path
        self.hitrate_path    = cfg.hitrate_path
        self.remained_b_path = cfg.remained_b_path

        #方便并行计算，配置用字典传输
        self.cfg_dict                    = {}
        self.cfg_dict["fetch_policy"]    = cfg.fetch_policy
        self.cfg_dict["if_noise"]        = cfg.if_noise
        self.cfg_dict["predict_policy"]  = cfg.predict_policy
        self.cfg_dict['PPF_delta']       = cfg.PPF_delta
        self.cfg_dict['slot_interval']   = cfg.slot_interval
        self.cfg_dict['update_interval'] = cfg.update_interval
        self.cfg_dict["c_e"]             = self.c_e
        self.cfg_dict["all_t"]           = cfg.all_t
        self.cfg_dict["test_day"]        = cfg.test_day
        self.cfg_dict['content_num']     = self.content_num
        self.cfg_dict["hyper_paras_num"] = self.hyper_paras_num

        # self.cfg_dict["xi"]              = cfg.xi
        # self.cfg_dict['PAC_D']           = cfg.PAC_D
        # self.cfg_dict["ifLoadKernel"]     = cfg.ifLoadKernel
        # self.cfg_dict['train_level']     = cfg.train_level
        # self.cfg_dict["Up_bound"]         = 100
        # self.cfg_dict["Low_bound"]        = 0.01
        # self.cfg_dict["B0"]               = self.B0


        # 评估指标

        self.total_hit   = np.zeros((self.all_t-self.test_begin_t,self.hyper_paras_num),dtype=np.int64)
        self.total_click = np.zeros(self.all_t-self.test_begin_t,dtype=np.int64)

    def init_new_eds(self, ed_indexs):
        if len(ed_indexs)>0:
            begintime = time.time()
            if self.predict_policy in ["HPP_PAC","HPP_SELF"]:
                pbar = tqdm(total=len(ed_indexs))
                pbar.set_description('Initializing EDs')
                update = lambda *args: pbar.update(1)
                pool = Pool(min(len(ed_indexs),self.maxProgressNum))#并行初始化
                results = []
            for ed_index in ed_indexs:
                if ed_index not in self.index2eds.keys():
                    if self.predict_policy in ["HPP_PAC","HPP_SELF"]:
                        eventArray = self.trace.get_ed_trace(ed_index).to_numpy(dtype=np.int32)
                        #global_parameter = self.ps.para_to_vector()
                        # parallel_para.append([])
                        results.append(pool.apply_async(init_ed_kernel, args=(ed_index, self.cfg_dict, eventArray,), callback=update))

                    ed = ED(events= self.trace.get_ed_trace(ed_index), index=ed_index, cfg = self.cfg_dict, content_num=self.trace.content_num)
                    self.index2eds[ed_index] = ed
            
            if self.predict_policy in ["HPP_PAC","HPP_SELF"]:
                pool.close()
                pool.join()
                pbar.close()
                for res in results:
                    ed = self.index2eds[res.get()[0]]
                    ed.HPP.setKernel(res.get()[1])
            elif self.predict_policy in ['HRS', "NO_MODEL","FUT","HIS","HIS_ONE"]:
                pass
            else:
                raise ValueError("not this predict_policy {}.".format({self.predict_policy}))


            print(f"Initialization finish! Total use time:{time.time()-begintime}")


    def set_t_o(self,t_o):
        self.ps.set_t_o(t_o)
        # for ed_index in self.index2eds.keys():
        #     ed = self.index2eds[ed_index]
        #     ed.HPP.set_t_o(t_o)

    def test_one_slot(self, slot_t):
        click_global = 0
        hit_global = np.zeros(self.hyper_paras_num,dtype=np.int64)

        for ed_index in self.index2eds.keys():  
            ed = self.index2eds[ed_index]

            #计算缓存命中率 / 无模型的缓存算法同时更新缓存
            ed_trace      = self.trace.get_ed_trace(ed_index)
            ed_slot_trace = ed_trace[ed_trace['time']==slot_t]
            ed_click, ed_hit =  ed.get_hitrate(ed_slot_trace) 
            click_global += ed_click
            hit_global   += ed_hit

            #计算强度函数
            if self.predict_policy in ["NO_MODEL"]:
                pass
            else:
                if self.predict_policy in ["HPP_SELF"]:
                    # beta = np.full(cfg_dict['content_num'],1)
                    # omega = np.full(cfg_dict['content_num'],1)
                    origin_density = self.ps.beta +  self.ps.omega * ed.HPP.sumKernel[:,slot_t]
                    # model_paras = [self.ps.beta,self.ps.omega, ed.HPP.sumKernel[:,slot_t]]
                elif self.predict_policy in ["HPP_PAC"]:
                    beta = self.ps.beta[ed_index,:]
                    omega = self.ps.p @ self.ps.q.T
                    sumkernel_t = ed.HPP.sumKernel[:,slot_t]
                    origin_density = beta + omega @ sumkernel_t
                    # model_paras = [self.ps.beta[ed_index,:],self.ps.p,self.ps.q,ed.HPP.sumKernel[:, slot_t]]
                elif self.predict_policy in ["FUT"]:
                    future_trace  = ed_trace[ed_trace['time']==slot_t+1]
                    content_indices = np.array(future_trace['i'])
                    future_count = np.bincount(content_indices,  minlength=self.content_num)
                    if future_trace.shape[0] != 0:
                        origin_density =  (future_count + 1) / future_trace.shape[0] 
                    else:
                        origin_density =  ed.density
                elif self.predict_policy in ["HIS"]:
                    origin_density = ed.content_count / ed.content_count.sum()
                    content_indices = np.array(ed_slot_trace['i'])
                    now_count = np.bincount(content_indices,  minlength=self.content_num)
                    ed.content_count = (1-0.1) * ed.content_count + 0.1 * now_count
                elif self.predict_policy in ["HIS_ONE"]:
                    content_indices = np.array(ed_slot_trace['i'])
                    now_count = np.bincount(content_indices,  minlength=self.content_num)
                    if ed_slot_trace.shape[0] != 0:
                        origin_density =  (now_count + 1) / ed_slot_trace.shape[0] 
                    else:
                        origin_density =  ed.density
                elif self.predict_policy in ["HRS"]:
                    sumK0 = ed.HPP.sumKernel[:,slot_t]
                    sumK1 = ed.HPP.sumNeKernel[:,slot_t]
                    density = ed.HPP.b + ed.HPP.omega * sumK0 - ed.HPP.gamma * sumK1
                    origin_density = np.where(density < 1e-8, 1e-9 * np.log(1 + 1e-15 + np.exp(density / 1e-9)), density)
                else:
                    raise ValueError("not this predict_policy {}.".format({self.predict_policy}))
                ed.update_density(origin_density)

            #计算相关性和更新加噪参数
            if self.if_noise:
                psi = ed.calculate_pearson_correlation()
            else:
                psi = None

            #更新缓存
            for index in range(self.hyper_paras_num):
                if self.fetch_policy in ["PPVF"]:
                    #获取不同缓存大小情况下的预请求
                    if self.test_hyper_paras == 'f_e':
                        redundant_request= ed.get_redundant_request(cache_index = index, f_e_total = self.f_e[index]*(ed_click-ed_hit[index]), epsilon = self.epsilon, xi=self.xi, psi=psi)
                    elif self.test_hyper_paras == 'c_e':
                        redundant_request= ed.get_redundant_request(cache_index = index, f_e_total = self.f_e*(ed_click-ed_hit[index]), epsilon = self.epsilon, xi=self.xi, psi=psi)
                    elif self.test_hyper_paras == 'epsilon':
                        redundant_request= ed.get_redundant_request(cache_index = index, f_e_total = self.f_e*(ed_click-ed_hit[index]), epsilon = self.epsilon[index], xi=self.xi, psi=psi)
                    elif self.test_hyper_paras == 'xi':
                        redundant_request= ed.get_redundant_request(cache_index = index, f_e_total = self.f_e*(ed_click-ed_hit[index]), epsilon = self.epsilon, xi=self.xi[index], psi=psi)
                    else:
                        raise ValueError(f'no this {self.test_hyper_paras}')
            
                    #更新缓存
                    if ed_click == 0 or ed_hit[index] == ed_click: #没有请求或全部本地命中
                        pass
                    elif ed_hit[index] < ed_click:
                        ed.update_cache(cache_index = index, redundant_request=redundant_request, ed_slot_trace = ed_slot_trace)
                    else: 
                        raise ValueError("running error...")
                else:
                    raise ValueError(f"not this fetch_policy {self.fetch_policy}.")
        return click_global, hit_global


    def main_Test(self):
        #---初始化---#
        all_trace = self.trace.get_all_trace()
        ed_indexs = set(all_trace[all_trace['time']<self.test_begin_t]['ed'])#获取新eds的index
        self.init_new_eds(ed_indexs)                                         #实例化新出现的eds

        #训练前准备
        self.t_o = self.test_begin_t
        self.set_t_o(self.t_o)                    #统一当前训练时间节点
        self.ps.update_eds_index(self.index2eds)  #更新client index
        self.update_predict_model()               #更新PS中HPP模型参数
        self.t_o += self.update_interval          #更新训练时间点
        self.test_one_slot(self.test_begin_t-1)   #初始化缓存策略，但不计入统计

        for slot_t in tqdm(range(self.test_begin_t, self.all_t)):
            new_ed_indexs = set(all_trace[all_trace['time']==slot_t]['ed']) - ed_indexs #获取新eds的index
            self.init_new_eds(new_ed_indexs)                                            #实例化新出现的eds
            ed_indexs = new_ed_indexs | ed_indexs                                       #更新完整的eds的index

            if slot_t >= self.t_o:
                self.set_t_o(self.t_o)                   #统一当前训练时间节点
                self.ps.update_eds_index(self.index2eds) #更新client index
                self.update_predict_model()              #更新PS中HPP模型参数
                self.t_o += self.update_interval         #更新训练时间点

            click_global_t, hit_global_t = self.test_one_slot(slot_t) #测试一个time slot
            self.total_hit[slot_t - self.test_begin_t,:] = hit_global_t
            self.total_click[slot_t - self.test_begin_t] = click_global_t
            self.update_writer_hitrate(slot_t - self.test_begin_t)
        
        self.store_final_state()

    def update_predict_model(self):
        if self.predict_policy in ["HPP_PAC","HPP_SELF"]:
            if self.run_way == "training":
                self.ps.global_fit_para()
                para_vector = self.ps.para_to_vector()
                # para_vector = res.x
                self.write_para(para_vector)
            else:
                para_vector = self.load_para()
                self.ps.vector_to_para(para_vector)
        elif self.predict_policy in ['HRS']:
            if self.run_way == "training":
                time0 = time.time()
                x = self.index2eds[0].HPP.para_to_x()
                para_vector = np.zeros((len(self.index2eds),x.shape[0]))
                # for ed_index in self.index2eds.keys():
                #     _ , self.index2eds[ed_index].HPP = HRS_online_traning(ed_index,self.index2eds[ed_index].HPP,onlineBeginT=self.t_o,maxiter=self.maxiter)

                pbar = tqdm(total=len(self.index2eds.keys()))
                pbar.set_description('Online training EDs')
                update = lambda *args: pbar.update(1)
                pool = Pool(min(len(self.index2eds.keys()),self.maxProgressNum))#并行初始化
                results = []
                for ed_index in self.index2eds.keys():
                    results.append(pool.apply_async(HRS_online_traning, args=(ed_index,self.index2eds[ed_index].HPP,self.t_o,self.maxiter), callback=update))
                pool.close()
                pool.join()
                pbar.close()
                for res in results:
                    ed = self.index2eds[res.get()[0]]
                    ed.HPP = res.get()[1]
                    para_vector[ed_index] = ed.HPP.para_to_x()
                print("online training use: ", time.time() - time0, 's\n')
                self.write_para(para_vector)
            else:
                para_vector = self.load_para()
                for ed_index in self.index2eds.keys():
                    self.index2eds[ed_index].HPP.x_to_para(para_vector[ed_index])
        elif self.predict_policy in ["NO_MODEL","FUT","HIS","HIS_ONE"]:
            pass
        else:
            raise ValueError("not this predict_policy {}.".format({self.predict_policy}))

    def write_para(self,para_vector):
        model_config_path = os.path.join(self.model_path, self.model_config)
        if not os.path.exists(model_config_path):
            os.makedirs(model_config_path)
        model_t_o_path = os.path.join(model_config_path, "%s.npy" % (self.t_o))
        para_vector = np.array(para_vector,dtype=np.float64)
        np.save(model_t_o_path, para_vector)

    def load_para(self):
        model_config_path = os.path.join(self.model_path, self.model_config)
        model_t_o_path = os.path.join(model_config_path, "%s.npy" % (self.t_o))
        if os.path.exists(model_t_o_path):
            para_vector = np.load(model_t_o_path)
        else:
            raise ValueError("no model file of predict_policy {}, and need train the model first.".format({self.predict_policy}))
        return para_vector

    #以下函数为tensorbroad写入相关函数
    def update_writer_hitrate(self, step):
        if self.test_hyper_paras in ['f_e']:
            for index,size in enumerate(self.f_e):
                # if self.total_click[step] != 0:
                #     self.summary_writer.add_scalar(
                #         f"hitrate_slot:{size}", self.total_hit[step,index]/self.total_click[step], step)
                # else:
                #     self.summary_writer.add_scalar(
                #         f"hitrate_slot:{size}", 1, step)
                if self.total_click.sum() != 0:
                    self.summary_writer.add_scalar(
                        f"hitrate_accum_fe:{size}", self.total_hit[:,index].sum()/self.total_click.sum(), step)
                else:
                    self.summary_writer.add_scalar(
                        f"hitrate_accum_fe:{size}", 1, step)
        elif self.test_hyper_paras in ['c_e']:
            for index,size in enumerate(self.c_e):
                # if self.total_click[step] != 0:
                #     self.summary_writer.add_scalar(
                #         f"hitrate_slot:{size}", self.total_hit[step,index]/self.total_click[step], step)
                # else:
                #     self.summary_writer.add_scalar(
                #         f"hitrate_slot:{size}", 1, step)
                if self.total_click.sum() != 0:
                    self.summary_writer.add_scalar(
                        f"hitrate_accum_ce:{size}", self.total_hit[:,index].sum()/self.total_click.sum(), step)
                else:
                    self.summary_writer.add_scalar(
                        f"hitrate_accum_ce:{size}", 1, step)
        elif self.test_hyper_paras in ['epsilon']:
            for index,size in enumerate(self.epsilon):
                # if self.total_click[step] != 0:
                #     self.summary_writer.add_scalar(
                #         f"hitrate_slot:{size}", self.total_hit[step,index]/self.total_click[step], step)
                # else:
                #     self.summary_writer.add_scalar(
                #         f"hitrate_slot:{size}", 1, step)
                if self.total_click.sum() != 0:
                    self.summary_writer.add_scalar(
                        f"hitrate_accum_eps:{size}", self.total_hit[:,index].sum()/self.total_click.sum(), step)
                else:
                    self.summary_writer.add_scalar(
                        f"hitrate_accum_eps:{size}", 1, step)
        elif self.test_hyper_paras in ['xi']:
            for index,size in enumerate(self.xi):
                # if self.total_click[step] != 0:
                #     self.summary_writer.add_scalar(
                #         f"hitrate_slot:{size}", self.total_hit[step,index]/self.total_click[step], step)
                # else:
                #     self.summary_writer.add_scalar(
                #         f"hitrate_slot:{size}", 1, step)
                if self.total_click.sum() != 0:
                    self.summary_writer.add_scalar(
                        f"hitrate_accum_xi:{size}", self.total_hit[:,index].sum()/self.total_click.sum(), step)
                else:
                    self.summary_writer.add_scalar(
                        f"hitrate_accum_xi:{size}", 1, step)
        self.summary_writer.add_scalar(
                "click_accum", self.total_click.sum(), step)


    def store_final_state(self):
        eds_final_actions  = np.zeros((len(self.index2eds),self.hyper_paras_num,self.content_num),dtype=np.int64)
        eds_final_remained_b = np.zeros((len(self.index2eds),self.hyper_paras_num,self.content_num))

        for ed_index in self.index2eds:
            ed = self.index2eds[ed_index]
            eds_final_actions[ed_index] = ed.actions
            eds_final_remained_b[ed_index] = ed.remained_b

        # 获取文件夹路径
        folder_path = os.path.dirname(self.actions_path)
        # 如果文件夹路径不存在，则创建它
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(f"{self.actions_path}",eds_final_actions)

        # 获取文件夹路径
        folder_path = os.path.dirname(self.remained_b_path)
        # 如果文件夹路径不存在，则创建它
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(f"{self.remained_b_path}",eds_final_remained_b)

        # 获取文件夹路径
        folder_path = os.path.dirname(self.hitrate_path)
        # 如果文件夹路径不存在，则创建它
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(f"{self.hitrate_path}",self.total_hit.sum(axis=0)/self.total_click.sum())