# encoding=utf-8
#!/usr/bin/env python
import sys
import numpy as np
import wave
import math
import matplotlib.pyplot as plt

import nextpow2    


def SpectralSubstraction( wavName ):
    
    ''' ================= read wav ===================='''
    f = wave.open( wavName )
    # 读取格式信息
    # (nchannels, sampwidth, framerate, nframes, comptype, compname)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    fs = framerate
    # 读取波形数据
    str_data = f.readframes(nframes)
    f.close()
    # 将波形数据转换为数组
    datax = np.fromstring(str_data, dtype=np.short)
     
    
    ''' ==================== set parameter =================='''
    leng = 20 * fs // 1000 #segment length
    PERC = 50 #overlap percent
    len1 = leng * PERC // 100 #overlap length
    len2 = leng - len1 #stride length
    
    # 设置默认参数
    Thres = 3
    Expnt = 2.0
    beta = 0.002
    G = 0.9
    
    
    ''' ====================== noise spetral ===================='''
    # 初始化汉明窗
    win = np.hamming(leng)
    # normalization gain for overlap+add with 50% overlap
    winGain = len2 / sum(win)
    
    # Noise magnitude calculations - assuming that the first 5 frames is noise/silence
    nFFT = 2 * ( 2 ** nextpow2.nextpow2(leng) )
    print('(len, nFFT): (%d, %d)' %(leng, nFFT))
    
    noise_mean = np.zeros(nFFT)
    j = 0
    for k in range(1, 6):
        noise_mean = noise_mean + abs(np.fft.fft(win * datax[j:j + leng], nFFT))
        j = j + leng
    noise_mu = noise_mean / 5


    
    # --- allocate memory and initialize various variables
    k = 1
    img = 1j
    x_old = np.zeros(len1)
    Nframes = len(datax) // len2 - 1 #(0, Nframes)
    print((Nframes, len(datax), len2))
    xfinal = np.zeros(Nframes * len2) #(0, Nframe*len2)
    
    print(datax)

   
    '''========================= Start Processing ============================'''
    for n in range(0, Nframes):
        # Windowing
        insign = win * datax[ (k-1) : (k + leng - 1)]
        # compute fourier transform of a frame
        spect = np.fft.fft(insign, nFFT)
        # compute the magnitude
        magni = abs(spect)

        # save the noisy phase information
        theta = np.angle(spect)
        SNRseg = 10 * np.log10(np.linalg.norm(magni, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

        def berouti(SNR):
            if -5.0 <= SNR <= 20.0:
                a = 4 - SNR * 3 / 20
            else:
                if SNR < -5.0:
                    a = 5
                if SNR > 20:
                    a = 1
            return a


        def berouti1(SNR):
            if -5.0 <= SNR <= 20.0:
                a = 3 - SNR * 2 / 20
            else:
                if SNR < -5.0:
                    a = 4
                if SNR > 20:
                    a = 1
            return a

    
        if Expnt == 1.0:  # 幅度谱
            alpha = berouti1(SNRseg)
        else:  # 功率谱
            alpha = berouti(SNRseg)    
        
        print(("SNRseg, alpha", SNRseg, alpha))
            
            
        #############
        sub_speech = magni ** Expnt - alpha * noise_mu ** Expnt;
        # 当纯净信号小于噪声信号的功率时
        diffw = sub_speech - beta * noise_mu ** Expnt
        # beta negative components

        def find_index(x_list):
            index_list = []
            for i in range(len(x_list)):
                if x_list[i] < 0:
                    index_list.append(i)
            return index_list

        z = find_index(diffw)

        if len(z) > 0:
            # 用估计出来的噪声信号表示下限值
            sub_speech[z] = beta * noise_mu[z] ** Expnt
            
            ''' ================= simple VAD: SNRseg < Thres =================='''
            if SNRseg < Thres:  # Update noise spectrum
                noise_temp = G * noise_mu ** Expnt + (1 - G) * magni ** Expnt  # 平滑处理噪声功率谱
                noise_mu = noise_temp ** (1 / Expnt)  # 新的噪声幅度谱
            # flipud函数实现矩阵的上下翻转，是以矩阵的“水平中线”为对称轴
            # 交换上下对称元素
            sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
            
            x_phase = (sub_speech ** (1 / Expnt)) * (np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta])))

            
            # take the IFFT
            xi = np.fft.ifft(x_phase).real
            # --- Overlap and add ---------------
            xfinal[k-1:k + len2 - 1] = x_old + xi[0:len1]
            x_old = xi[0 + len1:leng]
            k = k + len2

            
    ''' ========================= save file =============================='''
    # 保存文件
    wf = wave.open('en_outfile.wav', 'wb')
    # 设置参数
    wf.setparams(params)
    # 设置波形文件 .tostring()将array转换为data
    wave_data = (winGain * xfinal).astype(np.short)
    wf.writeframes(wave_data.tostring())
    wf.close()

    if(1):
        print('nchannels, sampwidth, framerate, nframes')
        print((nchannels, sampwidth, framerate, nframes))
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(range(len(datax)), datax)
        ax[1].plot(range(len(xfinal)), xfinal)
        plt.show()



if __name__ == "__main__":
    wavName = sys.argv[1]
    SpectralSubstraction( wavName )