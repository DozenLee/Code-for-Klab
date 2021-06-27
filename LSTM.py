#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import math
import sys
import time
import imutils
import numpy as np
import six

import chainer
from chainer import Link, Chain, ChainList, cuda, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import pickle
import json
import csv
from datetime import datetime
import os
import logging.config
import shutil
import lstm
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--test', '-t', default=0, type=int,
                    help='test')
parser.add_argument('--cont', '-c', default=0, type=int,
                    help='test')

args = parser.parse_args()



mod = cuda if args.gpu >= 0 else np
if args.gpu >= 0:
    print ("GPU ON")
    cuda.check_cuda_available()
    xp = cuda.cupy
else:
    print ("GPU OFF")
    xp = np



def mkdir(path):
    os.mkdir(path)
    print ("FILE", path, "OK")

def flatten(a):
    if not isinstance(a, (list, )):
        return [a]
    else:
        b = []
    for item in a:
        b += flatten(item)
    
    return b

SEQ_LENGTH = 10
class Train():
    def __init__(self):
        # 隠れ層の層数
        self.n_epoch = 5000
        self.n_skeleton_units = 10 # number of skeleton units
        self.n_ae_units = 1024  # number of auto encoder units
        self.batchsize = 20   # minibatch size default:20
        self.bprop_len = 40   # length of truncated BPTT（過去の状態をいくつまで残しておくか） default 35
        self.grad_clip = 1    # gradient norm threshold to clip（？） default:5
        self.seq_len = SEQ_LENGTH

        self.setting()

# ------------------------------------------------------------------- #
# setting(self)
# 内容 : 学習の環境を整える。やっていることは以下の通り
#          1.スケルトン・シーンのデータの読み込み
#          2.作業スペースを生成
#          3.作業スペースに学習データ・テストデータ・モデルファイルを保存
#          4.作業スペースに自身(このプログラム)を保存
# 引数 : self	  クラス自身
# 戻り値: なし
# ------------------------------------------------------------------- #
    def setting(self):
        # 学習に使用する各種データの読み込み
        # 訓練データのファイル名が書かれている設定ファイルを読み込む
        f_ske = open("configFiles/configTest_X.txt")
        filenameList_ske = f_ske.readlines()
        filename_train_X = 'dataset/' + filenameList_ske[0].replace('\n', '')
        filename_valid_X = 'dataset/' + filenameList_ske[1].replace('\n', '')
        filename_test_X  = 'dataset/' + filenameList_ske[2].replace('\n', '')

        f_ae = open("configFiles/configTest_Y.txt")
        filenameList_ae = f_ae.readlines()
        filename_train_Y = 'dataset/' + filenameList_ae[0].replace('\n', '')
        filename_varid_Y = 'dataset/' + filenameList_ae[1].replace('\n', '')
        filename_test_Y  = 'dataset/' + filenameList_ae[2].replace('\n', '')

        print ("train data name(ske): ", filename_train_X )      
        print ("train data name(ae) : ", filename_train_Y)

        # 訓練、バリ、テスト用のスケルトンデータ読み込み
        self.train_skeleton_data = self.load_data(filename_train_X, self.n_skeleton_units)
        print(filename_train_X)
        self.valid_skeleton_data = self.load_data(filename_valid_X, self.n_skeleton_units)
        self.test_skeleton_data = self.load_data(filename_test_X, self.n_skeleton_units) 

        # シーンのデータを読み込む
        self.train_ae_data = self.load_data(filename_train_Y, self.n_ae_units) 
        self.valid_ae_data = self.load_data(filename_varid_Y, self.n_ae_units)
        self.test_ae_data = self.load_data(filename_test_Y, self.n_ae_units)


        # プログラム開始時の日付時刻を取得
        self.rundate = datetime.now().strftime("%Y%m%d_%H%M%S")# + ("_test" if self.debug else "")
        # 日付時刻と使用した訓練データの名前を作業スペースとするのでその名前
        self.workspace = "resultTraining/actionSeq/" + self.rundate + "_" + filenameList_ske[0].split(".")[0].split("/")[0] #+ "_rawScene"
        # 作業スペースとしてフォルダを作成
        mkdir(self.workspace)

        # 作業スペースに使用した訓練データ、バリデータ、テストデータをコピー
        shutil.copyfile(filename_train_X, self.workspace + "/" + "train_X.txt")
        shutil.copyfile(filename_valid_X, self.workspace + "/" + "valid_X.txt")
        shutil.copyfile(filename_test_X, self.workspace + "/" + "test_X.txt")

        # 作業スペースに使用したAE中間層出力ファイルをコピー
        shutil.copyfile(filename_train_Y, self.workspace + "/" + "train_Y.txt")
        shutil.copyfile(filename_varid_Y, self.workspace + "/" + "varid_Y.txt")
        shutil.copyfile(filename_test_Y, self.workspace + "/" + "test_Y.txt")

        shutil.copyfile("lstm.py", self.workspace + "/" + "model.py")
        f_config = open(self.workspace + "/" + "config.txt", "w")
        f_config.write( "TrainSkeleton: "+ filename_train_X + "\n")
        f_config.write( "TrainScene   : "+ filename_train_Y + "\n")
        # 作業スペースに実行ファイルを保存する
        self.writeMyFile()

# ------------------------------------------------------------------- #
# load_data(self, filename, n_units)
# 内容 : 学習に使用するデータを読み込む
# 引数 : self	  クラス自身
#        filename 読み込むファイル名
#        n_units  データの1行(=時刻)あたりの次元数。
# 戻り値: datas	  読み込んだデータ
# ------------------------------------------------------------------- #
    def load_data(self, filename, n_units):
        f = open(filename)
        # 中間層の出力を格納
        datas = np.empty((0,n_units), dtype=np.float32)
        filename_main = filename.split(".")[0]
        for line in f.readlines():
            # 各行末尾に入っている改行を除く
            params = line.replace('\n', '').split(" ")
            #print(type(params),type(params[0]),params)
            data = np.array(params, dtype=np.float32)
            
            datas = np.vstack( [datas, data] )
        return datas

    # 使用したプログラムを作業フォルダにコピー
    def writeMyFile(self):
        shutil.copyfile(__file__, self.workspace + "/" + "prog.py")

# ------------------------------------------------------------------- #
# compute_loss(model, x1, x2, t, train=True)
# 内容 : モデルを用いてスケルトンのシーケンスを予測し、正解との平均二乗誤差を計算する
# 引数 : model	学習しているモデル
#        x1	    現時刻シーン
#        x2 	目標シーン
#	 t	正解データ(スケルトンのシーケンス)
#	 train	モデルの重みを更新するかどうか(デフォルト:True)
# 戻り値: F.mean_squared_error(y_,t_)  予測の平均二乗誤差
# ------------------------------------------------------------------- #
def compute_loss(model, x1, x2, t, train=True):
    if args.gpu >= 0:
        x1 = cuda.to_gpu(x1)
        x2 = cuda.to_gpu(x2)
        t = cuda.to_gpu(t)
    with chainer.using_config('enable_backprop', train):#参照　https://qiita.com/dsanno/items/286350ea5734a3543a0e
        x1_ = chainer.Variable(x1)
        x2_ = chainer.Variable(x2)
        x_ = F.concat(((x1_),(x2_)), axis=1)
        t_ = chainer.Variable(t)
        y_ = model(x_,Train=train)

    return F.mean_squared_error(y_,t_)

# ------------------------------------------------------------------- #
# outputLogFile(fp, list_log)
# 内容 : ログをファイルに出力する。配列形式で渡されたログをスペース区切りで出力
# 引数 : fp	  出力ファイルのポインタ
#        list_log 出力ログ(配列形式)
# 戻り値: なし
# ------------------------------------------------------------------- #
def outputLogFile(fp, list_log):
    l = len(list_log)
    for i in range(l-1):
        fp.write( str(list_log[i]) + " ")

    # 最後のみ別処理
    fp.write( str(list_log[l-1]) + "\n")

# ------------------------------------------------------------------- #
# evaluate(model, skeleton, scene, batchsize)
# 内容 : 学習中モデルの評価を行う。
# 引数 : model	  	学習しているモデル
#        skeleton	スケルトンデータ
#        scene 		シーン特徴
#	 batchsize	ミニバッチ数
# 戻り値: cuda.to_cpu(total_loss)/ske_len  1時刻毎の誤差
#        total_loss	  		  全時刻分の誤差の和
#
#        ※ total_lossはメインループ内で使用していない。メインループでは
#          誤差関数が複数項であることを想定してプログラムを作成しているが、
#          現在は誤差関数は1項であるため、数合わせのためにtotal_lossを
#          戻り値にしている。
# ------------------------------------------------------------------- #
def evaluate(model, skeleton, scene, batchsize):
    model.l2.h.volatile = 'on'
    model.l2.c.volatile = 'on'
    model.l3.h.volatile = 'on'
    model.l3.c.volatile = 'on'
        
    seq_len = SEQ_LENGTH
    ske_len = len(skeleton)
    scene_len = len(scene)
    jump = ske_len // batchsize
    total_loss = xp.zeros(())

    
    for i in six.moves.xrange(jump):

        cur_scene_batch = np.array([scene[(jump * j + i) % scene_len]
                                 for j in six.moves.xrange(batchsize)])
        target_scene_batch = np.array([scene[(jump * j + i + seq_len) % scene_len]
                                    for j in six.moves.xrange(batchsize)])

        teacher_ske_batch = np.empty((batchsize,0),dtype=np.float32)
        for seq in six.moves.xrange(seq_len):
            temp_batch = np.array([skeleton[(jump * j + i + seq) % ske_len]
                                   for j in six.moves.xrange(batchsize)])
            teacher_ske_batch = np.hstack([teacher_ske_batch, temp_batch])
            
        loss = compute_loss(model, cur_scene_batch, target_scene_batch, teacher_ske_batch, train=False)
        total_loss += loss.data.reshape(())

    return cuda.to_cpu(total_loss)/ske_len, total_loss # 手抜き    
 

###########################################
#
# 学習用関数
#
###########################################
def run():
    train = Train()
    model = lstm.LSTM(input_units=(train.n_ae_units*2), output_units=(train.n_skeleton_units*train.seq_len), hidden_units=500)
    if args.gpu >= 0:
        cuda.check_cuda_available()
        model.to_gpu()
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    # 学習状況出力用ファイル
    f_trainLog = open(train.workspace + "/" + "log_perp_train.txt", "w")
    f_testLog  = open(train.workspace + "/" + "log_perp_test.txt" , "w")
    # 累計誤差を格納する変数
    accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))
    ske_len = len(train.train_skeleton_data)
    ae_len = len(train.train_ae_data)
    jump = ske_len // train.batchsize
    print(ske_len,jump,ae_len)
    # モデルを保存するタイミングを調整する用の変数
    base = 1
    for epoch in six.moves.xrange(train.n_epoch + 1):
        model.reset_state()
        for i in six.moves.xrange(jump):
            cur_ae_batch = np.array([train.train_ae_data[(jump * j + i) % ae_len]
                                     for j in six.moves.xrange(train.batchsize)])
            target_ae_batch = np.array([train.train_ae_data[(jump * j + i + train.seq_len) % ae_len]
                                        for j in six.moves.xrange(train.batchsize)])
            teacher_ske_batch = np.empty((train.batchsize, 0),dtype=np.float32)
            for seq in six.moves.xrange(train.seq_len):
                temp_batch = np.array([train.train_skeleton_data[(jump * j + i + seq) % ske_len]
                                        for j in six.moves.xrange(train.batchsize)])
                teacher_ske_batch = np.hstack([teacher_ske_batch, temp_batch])
            accum_loss += compute_loss(model, cur_ae_batch, target_ae_batch, teacher_ske_batch, train=True)
            # lossに基づいてモデルの重み更新
            if (i + 1) % train.bprop_len == 0:
                optimizer.target.zerograds()
                accum_loss.backward() # ロス値の勾配を求める
                optimizer.update()

                accum_loss.unchain_backward()  # truncate (グラフから余分な部分を消し去る）
            # 累計誤差をリセットする
                accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))
                
        # epoch毎の処理
        # 端数分に対しても重み更新する
        optimizer.target.zerograds()
        accum_loss.backward() # ロス値の勾配を求める
        optimizer.update()
        accum_loss.unchain_backward()  # truncate (グラフから余分な部分を消し去る）
        # 累計誤差をリセットする
        accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))
        eva_trainSample_ = evaluate(copy.deepcopy(model), train.train_skeleton_data, train.train_ae_data, train.batchsize)
        eva_testSample_ = evaluate(copy.deepcopy(model), train.test_skeleton_data, train.test_ae_data, train.batchsize)
        # ログファイルに評価結果を出力
        outputLogFile(f_trainLog, eva_trainSample_)
        outputLogFile(f_testLog, eva_testSample_)
        #コンソールに評価結果を出力
        print ("epoch: ", epoch)
        print ("Train: loss_all: {:.9f}".format(eva_trainSample_[0]))
        print ("Test : loss_all: {:.9f}".format(eva_testSample_[0]))
        # モデルの保存
        if epoch % base == 0:
            #pickle.dump(self.model, open(self.workspace + "/" + 'model%04d_final' % (epoch), 'wb'), -1) # ネットワークモデルをファイルに保存
            if args.gpu >= 0:
                model.to_cpu()
            pickle.dump(model, open(train.workspace + "/" + 'model%04d_final' % (epoch), 'wb'), -1) # ネットワークモデルをファイルに保存
            serializers.save_npz(train.workspace + "/" + 'npz_model%04d_final' % (epoch),model)
            if args.gpu >= 0:
                model.to_gpu()                                                                 
        if epoch / base == 10:
            base *= 10
if __name__ == '__main__':
    print ("start")
    run()
