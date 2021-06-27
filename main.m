%%%%%%%%%%%  Speech processing  final project  
%written by hassan keshvari khojasteh
%2019
%%%%%%%%%%%%

clc
clear all
%%%%%% Training The Classifier %%%%%%%%%%%%%%%%

% import the clean audio signals

[s1,Fs]=audioread('F:\Projects\Master\Second Semester\Speech Processing\Final Project\Clean\ssp01.wav');
[s2,Fs]=audioread('F:\Projects\Master\Second Semester\Speech Processing\Final Project\Clean\ssp02.wav');
[s3,Fs]=audioread('F:\Projects\Master\Second Semester\Speech Processing\Final Project\Clean\ssp03.wav');
[s4,Fs]=audioread('F:\Projects\Master\Second Semester\Speech Processing\Final Project\Clean\ssp04.wav');
[s5,Fs]=audioread('F:\Projects\Master\Second Semester\Speech Processing\Final Project\Clean\ssp05.wav');
[s6,Fs]=audioread('F:\Projects\Master\Second Semester\Speech Processing\Final Project\Clean\ssp06.wav');

%import the clean noise signals 

[n1,Fs]=audioread('F:\Projects\Master\Second Semester\Speech Processing\Final Project\Noise\Bird.wav');
[n2,Fs]=audioread('F:\Projects\Master\Second Semester\Speech Processing\Final Project\Noise\Keyboard.wav');
[n3,Fs]=audioread('F:\Projects\Master\Second Semester\Speech Processing\Final Project\Noise\Restaurant.wav');
[n4,Fs]=audioread('F:\Projects\Master\Second Semester\Speech Processing\Final Project\Noise\Train.wav');
n1=(n1(:,1)+n1(:,2))/2;
n2=(n2(:,1)+n2(:,2))/2;
n3=(n3(:,1)+n3(:,2))/2;


S=s6;
N=n4;


% STFT of  signal 



             %%% framing the signal 
i=1;
j=1;
while  j  <= length(S)
    if  j+255 > length(S)
        S_W(:,i) =hamming(256).*(vertcat(S(j:length(S)),zeros(256-length(S(j:length(S))),1)));
    end
    if j+255 <= length(S)
      S_W(:,i) =hamming(256).*(S(j:255+j));
    end
    j=j+128;
    i=i+1;
end


                   %%framing the noise
i=1;
j=1;
while  j  <= length(N)
    if  j+255 > length(N)
        N_W(:,i) =hamming(256).*(vertcat(N(j:length(N)),zeros(256-length(N(j:length(N))),1)));
    end
    if j+255 <= length(N)
      N_W(:,i) =hamming(256).*(N(j:255+j));
    end
    j=j+128;
    i=i+1;
end

                %%%% calculating the fft of signal frames
                
v=size(S_W);
for i=1:v(2)
    ST_S(:,i)=fft(S_W(:,i));
end

                %%%% calculating the fft of noise frames

x=size(N_W);
for i=1:x(2)
    ST_N(:,i)=fft(N_W(:,i));
end
                %%%% calculating power specral density of signal & noise frames

STFT_S=abs(ST_S).^2;
STFT_N=abs(ST_N).^2;


%%%%%%%%% NMF & Weiner Filter Coeffcients %%%%%%%%%%%%
     

      
%My Implementation of Non_negative Matrix Factorization for speech signals
K=10;
v=size(STFT_S);
W_S=rand(v(1),K);
H_S=rand(K,v(2));

for i=1:1
    W_S=W_S.*(([STFT_S.*((W_S*H_S).^-1)]*H_S')./(ones(v)*H_S'));
    H_S=H_S.*((W_S'*[STFT_S.*((W_S*H_S).^-1)])./(W_S'*ones(v)));
    for l=1:K
        W_S(:,l)=W_S(:,l)./sum(abs(W_S(:,l)));
    end
    for l=1:K
        H_S(l,:)=H_S(l,:)./sum(abs(H_S(l,:)));
    end
end

%My Implementation of Non_negative Matrix Factorization for noise signals


u=size(STFT_N);
W_N=rand(u(1),K);
H_N=rand(K,u(2));

for i=1:1
    W_N=W_N.*(([STFT_N.*((W_N*H_N).^-1)]*H_N')./(ones(u)*H_N'));
    H_N=H_N.*((W_N'*[STFT_N.*((W_N*H_N).^-1)])./(W_N'*ones(u)));
    for l=1:K
        W_N(:,l)=W_N(:,l)./sum(abs(W_N(:,l)));
    end
    for l=1:K
        H_N(l,:)=H_N(l,:)./sum(abs(H_N(l,:)));
    end
end



                    %%%%%% calculating the weigner filter for signal parts
l_S=size(W_S);
J_S=W_S*H_S;
m_S=size(J_S);
G_S=zeros(l_S(1,2),m_S(1,1),m_S(1,2));

for i=1:l_S(1,2)
    G_S(i,:,:)=W_S(:,i)*H_S(i,:)./J_S;
    
end
                         %%%%%% calculating the weigner filter for noise parts


l_N=size(W_N);
J_N=W_N*H_N;
m_N=size(J_N);
G_N=zeros(l_N(1,2),m_N(1,1),m_N(1,2));

for i=1:l_N(1,2)
    G_N(i,:,:)=W_N(:,i)*H_N(i,:)./J_N;
end

%%%%%%%%Calculating the parts of signals & noises %%%%%%%%%%%

                  %%%Calculating the ifft of signal  parts
P_l_S=size(G_S);
P_S_F=zeros(P_l_S(1,1),P_l_S(1,2),P_l_S(1,3));

for i=1:P_l_S(1,1)
    P_S_F(i,:,:)=reshape(G_S(i,:,:),P_l_S(1,2),P_l_S(1,3)).*ST_S;
end

P_S_T=zeros(P_l_S(1,1),P_l_S(1,2),P_l_S(1,3));
for i=1:P_l_S(1,1)
    h_P_S=P_S_F(i,:,:);
    h_P_S=reshape(h_P_S,P_l_S(1,2),P_l_S(1,3));
    for k=1:P_l_S(1,3)
       P_S_T(i,:,k)=ifft(h_P_S(:,k));
    end
end

               %%deframing the signal parts
               
for i=1:P_l_S(1,1)
  P_S=reshape(P_S_T(i,:,:),P_l_S(1,2),P_l_S(1,3));  
  Q_S=P_S(:,1)./hamming(256);
  for j=2:P_l_S(1,3)
    M_S=P_S(:,j)./hamming(256);
    Q_S=vertcat(Q_S,M_S(129:256,1));
  end
  T_S(:,i)=Q_S;
end

               %%%Calculating the ifft of noise  parts

P_l_N=size(G_N);
P_N_F=zeros(P_l_N(1,1),P_l_N(1,2),P_l_N(1,3));

for i=1:P_l_N(1,1)
    P_N_F(i,:,:)=reshape(G_N(i,:,:),P_l_N(1,2),P_l_N(1,3)).*ST_N;
end

P_N_T=zeros(P_l_N(1,1),P_l_N(1,2),P_l_N(1,3));
for i=1:P_l_N(1,1)
    h_P_N=P_N_F(i,:,:);
    h_P_N=reshape(h_P_N,P_l_N(1,2),P_l_N(1,3));
    for k=1:P_l_N(1,3)
       P_N_T(i,:,k)=ifft(h_P_N(:,k));
    end
end


               %%deframing the noise parts

for i=1:P_l_N(1,1)
  P_N=reshape(P_N_T(i,:,:),P_l_N(1,2),P_l_N(1,3));  
  Q_N=P_N(:,1)./hamming(256);
  for j=2:P_l_N(1,3)
    M_N=P_N(:,j)./hamming(256);
    Q_N=vertcat(Q_N,M_N(129:256,1));
  end
  T_N(:,i)=Q_N;
end
            

% plot(S)
% figure
% d=max(abs(T_S(:,2)))
% plot(real(T_S(:,2)./d))
% 

%%%%%%% Calculating the MFCCs %%%%%%%%%%%  

Tw = 30;           % analysis frame duration (ms)
Ts = 10;           % analysis frame shift (ms)
alpha = 0.97;      % preemphasis coefficient
R = [0 8000 ];  % frequency range to consider
R1 = [0 22050 ];  % frequency range to consider
M = 20;            % number of filterbank channels 
C = 12;            % number of cepstral coefficients
L = 22;            % cepstral sine lifter parameter
      
% hamming window 
hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));



K=10
i=1;
for i=1:K
  [CC_S(i,:,:),FBC_S,frame_S(i,:,:)]=mfcc(T_S(:,i),16000,Tw,Ts,alpha,hamming,R,M,C,L);
  [CC_N(i,:,:),FBC_N,frame_N(i,:,:)]=mfcc(T_N(:,i),44100,Tw,Ts,alpha,hamming,R1,M,C,L);
  end

mfc_S=size(CC_S);
e_S=size(frame_S);
mfc_N=size(CC_N);
e_N=size(frame_N);

     
      %%averaging through signal MFCCs 

for i=1:mfc_S(1,1)
    for j=1:mfc_S(1,2)
        
        CC_S_(i,j,1)=mean(CC_S(i,j,:));
        
    end
end

    %%averaging through signal energy 

for i=1:e_S(1,1)
    for j=1:e_S(1,3)
        
        E_S(i,j)=(sum(frame_S(i,:,j).^2));
        
    end
end

for i=1:e_S(1,1)
           
        E_S_A(i)=log10(mean(E_S(i,:)));
end


      %%averaging through noise MFCCs


for i=1:mfc_N(1,1)
    for j=1:mfc_N(1,2)
        
        CC_N_(i,j,1)=mean(CC_N(i,j,:));
        
    end
end

  %%averaging through noise energy 

for i=1:e_N(1,1)
    for j=1:e_N(1,3)
        
        E_N(i,j)=(sum(frame_N(i,:,j).^2));
        
    end
end

for i=1:e_N(1,1)
           
        E_N_A(i)=log10(mean(E_N(i,:)));   
end


%concatenate the MFCCs anf log energy

CC_S_A=horzcat(CC_S_,E_S_A');
CC_N_A=horzcat(CC_N_,E_N_A');


%saving the extracted feature
CC_S_A6=CC_S_A;
CC_N_A1=CC_N_A;


save cc-a-s6.mat  CC_S_A6
save cc-a-n4.mat  CC_N_A1

%%%%%%% Training the SVM with Extracted features %%%%%%%%%%


X_train=vertcat(CC_S_A1,CC_S_A2,CC_S_A3,CC_S_A4,CC_S_A5,CC_S_A6,CC_N_A1,CC_N_A2,CC_N_A3,CC_N_A4);

l=6*size(CC_S_A1);
L_S=ones(l(1),1);

l=4*size(CC_N_A1);
L_N=-1.*ones(l(1),1);

Label=vertcat(L_S,L_N);

svm=fitcsvm(X_train,Label,'Standardize',true,'KernelFunction','RBF','ClassNames',[-1 1],'BoxConstraint',10);
cv = crossval(svm);
kfoldLoss(cv)


%%%%%%%%%% Enhancement %%%%%%%%%%%%%%%%%%%%%%

            %%% importing the noisy speech %%%%%%%%%%%%%%%%%%%%
            

audio_file=dir('F:\Projects\Master\Second Semester\Speech Processing\Final Project\10dB_train\*.wav');
n_audioes=length(audio_file);
  
S_Test=[]
for i=1:n_audioes
   currentadioname=audio_file(i).name;
   currentaudio=audioread(horzcat('F:\Projects\Master\Second Semester\Speech Processing\Final Project\10dB_train\',currentadioname));
  S_Test=vertcat(S_Test,currentaudio);
end

          %%%% STFT of noisy speech
          
i=1;
j=1;
while  j  <= length(S_Test)
    if  j+255 > length(S_Test)
        S_W_Test(:,i) =hamming(256).*(vertcat(S_Test(j:length(S_Test)),zeros(256-length(S_Test(j:length(S_Test))),1)));
    end
    if j+255 <= length(S_Test)
      S_W_Test(:,i) =hamming(256).*(S_Test(j:255+j));
    end
    j=j+128;
    i=i+1;
end
      
      %%%% fft of frames
      
v=size(S_W_Test);
for i=1:v(2)
    ST_Test(:,i)=fft(S_W_Test(:,i));
end

     %%% power spectral density

     
STFT_Test=abs(ST_Test).^2;

%%%%%%%%% NMF & Weiner Filter Coefitients %%%%%%%%%%%%
     
 %My Implementation of Non_negative Matrix Factorization for noise signals

K=10;
u_Test=size(STFT_Test);
W_Test=rand(u_Test(1),K);
H_Test=rand(K,u_Test(2));

for i=1:1
    W_Test=W_Test.*(([STFT_Test.*((W_Test*H_Test).^-1)]*H_Test')./(ones(u_Test)*H_Test'));
    H_Test=H_Test.*((W_Test'*[STFT_Test.*((W_Test*H_Test).^-1)])./(W_Test'*ones(u_Test)));
    for l=1:K
        W_Test(:,l)=W_Test(:,l)./sum(abs(W_Test(:,l)));
    end
    for l=1:K
        H_Test(l,:)=H_Test(l,:)./sum(abs(H_Test(l,:)));
    end
end


                 %%%%%% calculating the weigner filter for noisy speech  parts
                 
l_Test=size(W_Test);
J_Test=W_Test*H_Test;
m_Test=size(J_Test);
G_Test=zeros(l_Test(1,2),m_Test(1,1),m_Test(1,2));

for i=1:l_Test(1,2)
    G_Test(i,:,:)=W_Test(:,i)*H_Test(i,:)./J_Test;
end


%%%%%%%%Calculating the parts of signals & noises %%%%%%%%%%%

                  %%%Calculating the ifft of noisy speech  parts
P_l_Test=size(G_Test);
P_Test_F=zeros(P_l_Test(1,1),P_l_Test(1,2),P_l_Test(1,3));

for i=1:P_l_Test(1,1)
    P_Test_F(i,:,:)=reshape(G_Test(i,:,:),P_l_Test(1,2),P_l_Test(1,3)).*ST_Test;
end

P_Test_T=zeros(P_l_Test(1,1),P_l_Test(1,2),P_l_Test(1,3));
for i=1:P_l_Test(1,1)
    h_P_Test=P_Test_F(i,:,:);
    h_P_Test=reshape(h_P_Test,P_l_Test(1,2),P_l_Test(1,3));
    for k=1:P_l_Test(1,3)
       P_Test_T(i,:,k)=ifft(h_P_Test(:,k));
    end
end

               %%deframing the noisy speech parts
               
for i=1:P_l_Test(1,1)
  P_Test=reshape(P_Test_T(i,:,:),P_l_Test(1,2),P_l_Test(1,3));  
  Q_Test=P_Test(:,1)./hamming(256);
  for j=2:P_l_Test(1,3)
    M_Test=P_Test(:,j)./hamming(256);
    Q_Test=vertcat(Q_Test,M_Test(129:256,1));
  end
  T_Test(:,i)=Q_Test;
end

  
%%%%%%% Calculating the MFCC of noisy speech %%%%%%%%%%%  

Tw = 30;           % analysis fr ame duration (ms)
Ts = 10;           % analysis frame shift (ms)
alpha = 0.97;      % preemphasis coefficient
R = [0 22050 ];  % frequency range to consider
M = 20;            % number of filterbank channels 
C = 12;            % number of cepstral coefficients
L = 22;            % cepstral sine lifter parameter
      
% hamming window 
hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));


i=1;
for i=1:K
  [CC_Test(i,:,:),FBC_Test,frame_Test(i,:,:)]=mfcc(T_Test(:,i),44100,Tw,Ts,alpha,hamming,R,M,C,L);
end

e_Test=size(frame_Test);
mfc_Test=size(CC_Test);

       

      %%averaging through noisy speech MFCCs 

for i=1:mfc_Test(1,1)
    for j=1:mfc_Test(1,2)
        
        CC_Test_(i,j,1)=mean(CC_Test(i,j,:));
        
    end
end

  %%averaging through noisy speech Energy 
  
for i=1:e_Test(1,1)
    for j=1:e_Test(1,3)
        
        E_Test(i,j)=(sum(frame_Test(i,:,j).^2));
        
    end
end

for i=1:e_Test(1,1)
           
        E_Test_A(i)=log10(mean(E_Test(i,:)));   
end

%concatenate the MFCCs anf log energy

CC_Test_A=horzcat(CC_Test_,E_Test_A');

%predict the noisy speech parts
  
 save cc-a-test.mat  CC_Test_A

 [label,score]=predict(svm,CC_Test_A);
    
    
    %%%%% sum the signal parts of noisy speech %%%%%%%%%
    
    S_R=zeros(length(T_Test(:,1)),1);
    for i=1:length(label)
        l=label(i);
        if l==1
            S_R=horzcat(S_R,T_Test(:,i));
        end
    end
    
    S_R=sum(S_R,2);
    
    
    
    %%%% Compute the PESQ %%%%%%%%%
    
    %Change the  Sampling Frequency of clean speech to become equal to the Sampling Frequency of noisy speech
    
d_u=interp(s1,441); 
d_d=decimate(d_u,160);
    
[PESQ,MOSQ]=pesqbin(d_d',S_R(1:length(d_d))',Fs,'nb');

    
   % %%%%Compute SDR  of noisy speech and Enhanced noisy speech%%%%%%%%%%%
    


[SDR_Enhanced,ISR,SIR,SAR,perm]=bss_eval_images(S_R(1:length(d_d))',d_d');  
