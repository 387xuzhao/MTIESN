clc;
clear all;                                            
t1=clock; Time=[]; 
rmse_train1 = [];  mape_train1=[];    rmse_test1=[];  mape_test1=[];
rmse_train2 = [];  mape_train2=[];    rmse_test2=[];  mape_test2=[];
for L=1 
  rand('state',sum(100*clock));   t1=clock;     
  Loss_train=[]; Loss_test=[];
  traintarget1=[];   trainshuchu1=[];   Errortrain1=[]; Loss_train1=[];
  testtarget1=[];    testshuchu1=[];    Errortest1=[];  Loss_test1=[];
  traintarget2=[];   trainshuchu2=[];   Errortrain2=[]; Loss_train2=[]; 
  testtarget2=[];    testshuchu2=[];    Errortest2=[];  Loss_test2=[];
%%%%%%%%%%%%%%%%%%Read input and output sequences 
   inputrawData1 = xlsread('Rawdata.xlsx',1,'E2:G43201');   
   outputrawData1 = xlsread('Rawdata.xlsx',2,'B2:B43201');  %PE
   inputrawData2 =  xlsread('Rawdata.xlsx',1,'L2:N43201');
   outputrawData2 = xlsread('Rawdata.xlsx',2,'C2:C43201');  %AE
   inputrawData3 = xlsread("Rawdata.xlsx",1,'C2:D43201');
   inputrawData4 = xlsread("Rawdata.xlsx",1,'J2:K43201');
   %%%%%%%%%%%%%%%%%%%%%%% data interpolation
   totalNum1 = size(inputrawData1(1:60:end,:),1);  totalNum2 = size(inputrawData2(1:60:end,:),1);
   tempIn1 = zeros(1,size(inputrawData1,2)); inputseries1 = [];labeledX1=[]; 
   tempOut1 = zeros(1,size(outputrawData1,2)); outputseries1 = [];labeledY1=[]; inputseries3 =[]; labeledX3=[];
   n=59;  zuijinlin1 =24;  zuijinlin2 =24;  
   tempIn2 = zeros(1,size(inputrawData2,2)); inputseries2 = [];labeledX2=[];
   tempOut2 = zeros(1,size(outputrawData2,2)); outputseries2 = [];labeledY2=[]; inputseries4 =[]; labeledX4=[];
for i = 1 : totalNum1   
    index1 = 60 * (i-1);  inputseries1 = [inputseries1;inputrawData1(index1+1,:)]; 
    outputseries1 = [outputseries1;outputrawData1(index1+1,:)];   t=1;
    inputseries3 = [inputseries3; inputrawData3(index1+1,:)];
    if length(labeledX1) <= zuijinlin1
        labeledX1=[labeledX1;inputrawData1(index1+1,:)];   
        labeledY1=[labeledY1;outputrawData1(index1+1,:)];  
        labeledX3 =[labeledX3;inputrawData3(index1+1,:)];
    end
    if length(labeledX1) > zuijinlin1
      labeledX1=[labeledX1(t+1:t+zuijinlin1-1,:);inputrawData1(index1+1,:)];
      labeledY1=[labeledY1(t+1:t+zuijinlin1-1,:);outputrawData1(index1+1,:)];
      labeledX3 =[labeledX3(t+1:t+zuijinlin1-1,:);inputrawData3(index1+1,:)];
      t=t+1;
    end
    for j = 2:60
       tempIn1 = tempIn1 + inputrawData1(index1 + j , :);
    end
    inputseries1 = [inputseries1;tempIn1./n];
unlabeledX1 =tempIn1./n;   distancezong1 = 1; 
weights1 = zeros(size(labeledX1, 1), 1); 
for k = 1:size(labeledX1, 1)
    distance1 = pdist2(unlabeledX1(1, :), labeledX1(k, :), 'euclidean');
    distancezong1 = [distancezong1;distance1];
    weights1(k) = (distance1 - min(distancezong1))/(max(distancezong1) -min(distancezong1));  
end
normalizedWeights1 = weights1 / sum(weights1);  prediction1 = normalizedWeights1' * labeledY1;
prediction3 = normalizedWeights1' * labeledX3; tempIn1 = 0;   
   if isnan(prediction1)
       prediction1 = outputrawData1(index1+1,:);
       prediction3 = inputrawData3(index1+1,:);
   end
outputseries1=[outputseries1;prediction1];
inputseries3 =[inputseries3; prediction3];
end  
for i = 1 : totalNum2  
    index2 = 60 * (i-1);  inputseries2 = [inputseries2;inputrawData2(index2+1,:)];  
    outputseries2 = [outputseries2;outputrawData2(index2+1,:)];  
    inputseries4 = [ inputseries4 ; inputrawData4(index2+1,:)];
    t=1;
    if length(labeledX2) <= zuijinlin2
        labeledX2=[labeledX2;inputrawData2(index2+1,:)];   
        labeledY2=[labeledY2;outputrawData2(index2+1,:)]; 
        labeledX4 =[labeledX4; inputrawData4(index2+1,:)];
    end
    if length(labeledX2) > zuijinlin2
      labeledX2=[labeledX2(t+1:t+zuijinlin2-1,:);inputrawData2(index2+1,:)];
      labeledY2=[labeledY2(t+1:t+zuijinlin2-1,:);outputrawData2(index2+1,:)];
      labeledX4=[labeledX4(t+1:t+zuijinlin2-1,:);inputrawData4(index2+1,:)];
      t=t+1;
    end
    for j = 2:60 
       tempIn2 = tempIn2 + inputrawData2(index2 + j , :);
    end
    inputseries2 = [inputseries2;tempIn2./n];
unlabeledX2 =tempIn2./n;   distancezong2 = 1;
weights2 = zeros(size(labeledX2, 1), 1); 
for k = 1:size(labeledX2, 1)
     distance2 = pdist2(unlabeledX2(1, :), labeledX2(k, :), 'euclidean');
     distancezong2 = [distancezong2;distance2];
     weights2(k) = (distance2 - min(distancezong2))/(max(distancezong2) -min(distancezong2));
end
normalizedWeights2 = weights2 / sum(weights2); prediction2 = normalizedWeights2' * labeledY2; 
prediction4 = normalizedWeights2' * labeledX4;   tempIn2 = 0;
   if isnan(prediction2)
       prediction2 = outputrawData2(index2+1,:);
       prediction4 = inputrawData4(index2+1,:);
   end
 outputseries2 = [outputseries2;prediction2];
 inputseries4  = [ inputseries4;prediction4];
end  
%%%%%%%%%%%%%%%%%%%%%%%%%Obtain sequence S
inputData1= [inputseries3, inputseries1];   outputData1 = outputseries1;
inputData2= [inputseries4, inputseries2];   outputData2 = outputseries2;
%%%%%%%%%%%%%%%%%%%%Initialize the model parameters
N_sub = 10;    N_1 = N_sub;  nInternalUnits1 = N_1;  N_2 = N_sub;  nInternalUnits2 = N_2; 
Singular1=0.1+(0.99-0.1)*rand(1,nInternalUnits1);   Singular2=0.1+(0.99-0.1)*rand(1,nInternalUnits2); 
%%%%%%%%%%%%%%%%%%%%Eliminate outliers and normalize
output1 = fun_remove_outlier(outputData1); 
[output1, PS1] = mapminmax(output1',0,1);  output1 = output1';
input1 = mapminmax(inputData1',0,1);    input1 = input1';
output2 = fun_remove_outlier(outputData2); 
[output2, PS2] = mapminmax(output2',0,1);  output2 = output2';
input2 = mapminmax(inputData2',0,1);    input2 = input2';
nInputUnits1 = size(input1,2);  nOutputUnits1 = 1; 
inputWeights1 = 0.2 * rand(nInternalUnits1, nInputUnits1)- 0.1;     
outputbackWeights1=0.2 * rand(nInternalUnits1, nOutputUnits1)- 0.1;  
DiagonalMatrix1=diag(Singular1);  U1=rands(nInternalUnits1,nInternalUnits1);    
V1=rands(nInternalUnits1,nInternalUnits1);  OrthU1=orth(U1);  OrthV1=orth(V1);
internalWeights1=OrthU1*DiagonalMatrix1*OrthV1;
nInputUnits2 = size(input2,2);  nOutputUnits2 = 1; 
inputWeights2 = 0.2 * rand(nInternalUnits2, nInputUnits2)- 0.1;     
outputbackWeights2=0.2 * rand(nInternalUnits2, nOutputUnits2)- 0.1;  
DiagonalMatrix2=diag(Singular2); U2=rands(nInternalUnits2,nInternalUnits2);   
V2=rands(nInternalUnits2,nInternalUnits2); OrthU2=orth(U2); OrthV2=orth(V2);
internalWeights2=OrthU2*DiagonalMatrix2*OrthV2;
outputWeights1 = zeros(length(internalWeights1)+length(internalWeights2),nOutputUnits1);
outputWeights2 = zeros(length(internalWeights1)+length(internalWeights2),nOutputUnits2);
outputWeightszong = [outputWeights1,outputWeights2];
ntrainForgetPoints=0; ntestForgetPoints=0;   
trainsample=480*2;   
testsample=240*2;     
r=0.1;  contributions=[];
Ea_gen = 250;    Ea = 0.5 * Ea_gen^2 ; 
gamma = 0.05;   max_reservoir_size = 200;
%%%%%%%%%%%%%%%%%%%%Divide training and test data sets
[traininput1,  testinput1]=split_train_test(input1, trainsample, testsample);
[trainoutput1, testoutput1]=split_train_test(output1, trainsample, testsample);
[traininput2,  testinput2]=split_train_test(input2, trainsample, testsample);
[trainoutput2, testoutput2]=split_train_test(output2, trainsample, testsample);
%%%%%%%%%%%%%%%%%%%%%%%%model training
for xulie=1:trainsample
   trainInputSequence1=traininput1(xulie,:);    trainOutputSequence1=trainoutput1(xulie,:);
   trainInputSequence2=traininput2(xulie,:);    trainOutputSequence2=trainoutput2(xulie,:);
  TrainstateMatrix_train = compute_twoshare(trainInputSequence1,trainInputSequence2,trainOutputSequence1,trainOutputSequence2, ...
        ntrainForgetPoints,nInternalUnits1,nInternalUnits2, nOutputUnits1,nOutputUnits2, ...
      internalWeights1,internalWeights2,inputWeights1,inputWeights2,outputbackWeights1,outputbackWeights2);
 teacher_trainguiyi = compute_twoteacher(trainOutputSequence1,trainOutputSequence2,ntrainForgetPoints);
 Trainteacherguiyi1= teacher_trainguiyi(:,1);  Trainteacherguiyi2= teacher_trainguiyi(:,2);  
  for diedai = 1:25
       yucetrainguiyizong = TrainstateMatrix_train * outputWeightszong;
       error_trainzong = yucetrainguiyizong - teacher_trainguiyi;
       gradient_W_out = TrainstateMatrix_train' * error_trainzong;
       outputWeightszong = outputWeightszong - r*gradient_W_out;
  end
    outputWeights1 = outputWeightszong(:,1);  outputWeights2 = outputWeightszong(:,2);
    yucetrainguiyi1 = yucetrainguiyizong(1,1);  yucetrainguiyi2 = yucetrainguiyizong(1,2);
    yucetrain1 = mapminmax('reverse',yucetrainguiyi1',PS1);    yucetrain1 = yucetrain1';
    shijitrain1 = mapminmax('reverse',Trainteacherguiyi1',PS1);    shijitrain1 = shijitrain1';
    traintarget1=[traintarget1,shijitrain1(end,:)];    trainshuchu1=[trainshuchu1,yucetrain1];
    yucetrain2 = mapminmax('reverse',yucetrainguiyi2',PS2);    yucetrain2 = yucetrain2';
    shijitrain2 = mapminmax('reverse',Trainteacherguiyi2',PS2);    shijitrain2 = shijitrain2';
    traintarget2=[traintarget2,shijitrain2(end,:)];    trainshuchu2=[trainshuchu2,yucetrain2];
    error_train1 = yucetrain1 -shijitrain1;  Errortrain1 = [Errortrain1;error_train1];
    error_train2 = yucetrain2 -shijitrain2;  Errortrain2 = [Errortrain2;error_train2];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%Modular growth strategy
  train_loss1 = 0.5*error_train1.^2;    train_loss2 = 0.5*error_train2.^2;  
  Loss_train1 = [Loss_train1;train_loss1];  Loss_train2 = [Loss_train2;train_loss2];  
  train_loss =train_loss1 + train_loss2;   Loss_train = [Loss_train; train_loss];  % Integrated modeling error
  num_subreservoirs=length(outputWeightszong)/N_sub;
  for d = 1:num_subreservoirs  % Calculate contribution for each sub-reservoir
        sub_reservoir_start = (d-1)*N_sub + 1;     sub_reservoir_end = d*N_sub; 
        phi1_d = (TrainstateMatrix_train(1,sub_reservoir_start:sub_reservoir_end) * outputWeightszong(sub_reservoir_start:sub_reservoir_end,1) ) / ...
            sum(TrainstateMatrix_train(1,:) * outputWeightszong(:,1));  % Task 1 contribution
        phi2_d = (TrainstateMatrix_train(1,sub_reservoir_start:sub_reservoir_end) * outputWeightszong(sub_reservoir_start:sub_reservoir_end,2)) / ...
            sum(TrainstateMatrix_train(1,:) * outputWeightszong(:,2)); % Task 2 contribution
        contributions(d, 1) = phi1_d;        contributions(d, 2) = phi2_d;
  end
  if train_loss > Ea && length(outputWeightszong) <= 2*max_reservoir_size   % Check if error exceeds the threshold
      cum_contributions = sum(contributions, 2);  
      module_contributions(1,1) = sum(cum_contributions(1:nInternalUnits1/N_sub,1));
      module_contributions(2,1) = sum(cum_contributions(nInternalUnits1/N_sub+1:(nInternalUnits1+nInternalUnits2)/N_sub,1));
      [~, alpha] = min(abs(module_contributions)); 
      if alpha==1 && nInternalUnits1 < max_reservoir_size 
       [~, alpha_h] = max(abs(cum_contributions(1:nInternalUnits1/N_sub,1)));  
         inputWeights_new1 = inputWeights1(alpha_h*N_sub-N_sub+1:alpha_h*N_1,:); internalWeights_new1=internalWeights1(alpha_h*N_sub-N_sub+1:alpha_h*N_1,alpha_h*N_sub-N_sub+1:alpha_h*N_1);
         outputbackWeights1_new = outputbackWeights1(alpha_h*N_sub-N_sub+1:alpha_h*N_1,1);
       TrainstateMatrix_new1 = compute_statematrix(trainInputSequence1,trainOutputSequence1,...
                     ntrainForgetPoints,N_1,nOutputUnits1,internalWeights_new1,inputWeights_new1,outputbackWeights1_new);
         outputWeights_newzong1=(teacher_trainguiyi'-yucetrainguiyizong')/TrainstateMatrix_new1';   
        outputWeights1=[outputWeights1(1:nInternalUnits1,1);outputWeights_newzong1(1,:)';outputWeights1(nInternalUnits1+1:end,1)];
        outputWeights2=[outputWeights2(1:nInternalUnits1,1);outputWeights_newzong1(2,:)';outputWeights2(nInternalUnits1+1:end,1)];
        outputWeightszong = [outputWeights1,outputWeights2];
        internalWeights1 = [internalWeights1, zeros(nInternalUnits1,N_1);
                            zeros(N_1,nInternalUnits1), internalWeights_new1];
        inputWeights1 = [inputWeights1 ; inputWeights_new1];     outputbackWeights1 = [outputbackWeights1 ; outputbackWeights1_new];
        TrainstateMatrix_train = [TrainstateMatrix_train(:,1:nInternalUnits1),TrainstateMatrix_new1,TrainstateMatrix_train(:,nInternalUnits1+1:end)];
        nInternalUnits1=nInternalUnits1+N_1; 
      end
      if alpha==2 && nInternalUnits2 < max_reservoir_size     
       [~, alpha_h] = max(abs(cum_contributions(nInternalUnits1/N_sub+1:(nInternalUnits1+nInternalUnits2)/N_sub,1)));  % Identify the sub-reservoir with the maximum contribution in module alpha
        inputWeights_new2 = inputWeights2(alpha_h*N_sub-N_sub+1:alpha_h*N_1,:);internalWeights_new2=internalWeights2(alpha_h*N_sub-N_sub+1:alpha_h*N_1,alpha_h*N_sub-N_sub+1:alpha_h*N_1);
        outputbackWeights2_new = outputbackWeights2(alpha_h*N_sub-N_sub+1:alpha_h*N_1,1);
        TrainstateMatrix_new2 = compute_statematrix(trainInputSequence2,trainOutputSequence2,...
                        ntrainForgetPoints,N_2,nOutputUnits2,internalWeights_new2,inputWeights_new2,outputbackWeights2_new);
        outputWeights_newzong2=(teacher_trainguiyi'-yucetrainguiyizong')/TrainstateMatrix_new2';    
        outputWeights1=[outputWeights1(1:nInternalUnits1+nInternalUnits2,1);outputWeights_newzong2(1,:)'];
        outputWeights2=[outputWeights2(1:nInternalUnits1+nInternalUnits2,1);outputWeights_newzong2(2,:)'];
        outputWeightszong = [outputWeights1,outputWeights2];
        internalWeights2 = [internalWeights2, zeros(nInternalUnits2,N_2);
                           zeros(N_2,nInternalUnits2), internalWeights_new2];
        inputWeights2 = [inputWeights2 ; inputWeights_new2]; outputbackWeights2 =[outputbackWeights2;outputbackWeights2_new];
        TrainstateMatrix_train = [TrainstateMatrix_train(:,1:nInternalUnits1+nInternalUnits2),TrainstateMatrix_new2];
        nInternalUnits2=nInternalUnits2+N_2; 
      end
  end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%Module deletion strategy
  cum_contributions = sum(contributions, 2);
  [~, beta_h] = min(abs(cum_contributions));
  cum_contributions2 = [cum_contributions(1:beta_h-1,1) ; cum_contributions(beta_h+1:end,1)];
  [~, beta_h2] = min(abs(cum_contributions2));
  inputWeightszong = [inputWeights1;inputWeights2];
  internalWeightszong = [internalWeights1, zeros(nInternalUnits1,nInternalUnits2);
                        zeros(nInternalUnits2,nInternalUnits1), internalWeights2];
  if cum_contributions(beta_h,1) < gamma  
    outputWeightszong(beta_h2*N_1-N_1+1:beta_h2*N_1,:) = outputWeightszong(beta_h2*N_1-N_1+1:beta_h2*N_1,:)...
           + outputWeightszong(beta_h*N_1-N_1+1:beta_h*N_1,:) * (TrainstateMatrix_train(1,beta_h*N_1-N_1+1:beta_h*N_1) / TrainstateMatrix_train(1,beta_h2*N_1-N_1+1:beta_h2*N_1));
     outputWeightszong(beta_h*N_1-N_1+1:beta_h*N_1,:) = 0;
     inputWeightszong(beta_h*N_1-N_1+1:beta_h*N_1,:) = 0;
     internalWeightszong(beta_h*N_1-N_1+1:beta_h*N_1,beta_h*N_1-N_1+1:beta_h*N_1)= 0;
  end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%model test
for xulie = 1:testsample
    testInputSequence1=testinput1(xulie,:);  testOutputSequence1=testoutput1(xulie,:);
    testInputSequence2=testinput2(xulie,:);  testOutputSequence2=testoutput2(xulie,:);
   if xulie ==1 || xulie == 2
        testOutputSequence11=testoutput1(xulie,:);
        testOutputSequence22=testoutput2(xulie,:);
    else
        testOutputSequence11=testoutput1(xulie-2,:);
        testOutputSequence22=testoutput2(xulie-2,:);
   end
    stateCollection = compute_twoshare(testInputSequence1,testInputSequence2,testOutputSequence1,testOutputSequence2, ...
        ntestForgetPoints,nInternalUnits1,nInternalUnits2,nOutputUnits1,nOutputUnits2, ...
      internalWeights1,internalWeights2,inputWeights1,inputWeights2,outputbackWeights1,outputbackWeights2);
    TeachOutputSequence = compute_twoteacher(testOutputSequence1,testOutputSequence2,ntestForgetPoints);
    TeachOutputguiyi1= TeachOutputSequence(:,1);  TeachOutputguiyi2= TeachOutputSequence(:,2); 
     TeachOutputSequence00 = compute_twoteacher(testOutputSequence11,testOutputSequence22,ntestForgetPoints);
       for diedai = 1:40
             yucetestguiyizong = stateCollection * outputWeightszong;
             error_testzong = yucetestguiyizong - TeachOutputSequence00;
             gradient_W_out = stateCollection' * error_testzong;
             outputWeightszong = outputWeightszong - r*gradient_W_out;
      end
            outputWeights1 = outputWeightszong(:,1);outputWeights2 = outputWeightszong(:,2);
            yucetestguiyi1 = yucetestguiyizong(1,1);yucetestguiyi2 = yucetestguiyizong(1,2);
    yucetest1 = mapminmax('reverse',yucetestguiyi1',PS1);    yucetest1 = yucetest1';
    shijitest1 = mapminmax('reverse',TeachOutputguiyi1',PS1);    shijitest1 = shijitest1';
    testtarget1=[testtarget1,shijitest1];    testshuchu1=[testshuchu1,yucetest1];
    errortest1 = testshuchu1(xulie) - testtarget1(xulie);   Errortest1 = [Errortest1 errortest1];
    yucetest2 = mapminmax('reverse',yucetestguiyi2',PS2);    yucetest2 = yucetest2';
    shijitest2 = mapminmax('reverse',TeachOutputguiyi2',PS2);    shijitest2 = shijitest2';
    testtarget2=[testtarget2,shijitest2];    testshuchu2=[testshuchu2,yucetest2];
    errortest2 = testshuchu2(xulie) - testtarget2(xulie); Errortest2 = [Errortest2 errortest2];
end
%%%%%%%%%%%%%%%%%%%%%Take the sequence S1 of the real moment
traintarget1= traintarget1(1,1:2:end);     trainshuchu1 = trainshuchu1(1,1:2:end); 
traintarget2= traintarget2(1,1:2:end);     trainshuchu2 = trainshuchu2(1,1:2:end); 
testtarget1 = testtarget1(1,1:2:end);      testshuchu1  = testshuchu1(1,1:2:end); 
testtarget2 = testtarget2(1,1:2:end);      testshuchu2  = testshuchu2(1,1:2:end); 
Errortest1 = Errortest1(1,1:2:end);        Errortest2 = Errortest2(1,1:2:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%plot curves
figure(1);  plot(testtarget1,'g-');  
hold on;    plot(testshuchu1,'r-');
axis tight;   xlabel('Step \it t');
legend('Actural','MTIESN');   title('PE');   hold off;
figure(2);  plot(testtarget2,'g-');  
hold on;    plot(testshuchu2,'r-');
axis tight;   xlabel('Step \it t');
legend('Actural','MTIESN');   title('AE');   hold off;
%%%%%%%%%%calculate the RMSE and MAPE of training and testing for PE and AE
nEstimatePointstrain1=length(traintarget1); TeachVariancetrain1 = var(traintarget1) ;                                     
meanerrortrain1 = sum((trainshuchu1 - traintarget1).^2)/nEstimatePointstrain1; 
NRMSE_train1 = (sqrt(meanerrortrain1./TeachVariancetrain1));                    
RMSE_train1=sqrt(meanerrortrain1);
MAPE_train1 = sum(abs((trainshuchu1-traintarget1)./traintarget1))/nEstimatePointstrain1;
nEstimatePointstrain2=length(traintarget2); TeachVariancetrain2 = var(traintarget2) ;                                     
meanerrortrain2 = sum((trainshuchu2 - traintarget2).^2)/nEstimatePointstrain2; 
NRMSE_train2 = (sqrt(meanerrortrain2./TeachVariancetrain2));                    
RMSE_train2=sqrt(meanerrortrain2);
MAPE_train2 = sum(abs((trainshuchu2-traintarget2)./traintarget2))/nEstimatePointstrain2;
nEstimatePointstest1=length(testshuchu1);   TeachVariancetest1 = var(testshuchu1);                                 
meanerrortest1 = sum((testshuchu1 - testtarget1).^2)/nEstimatePointstest1;  
NRMSE_test1 = (sqrt(meanerrortest1./TeachVariancetest1));           
RMSE_test1=sqrt(meanerrortest1);
MAPE_test1 = sum(abs((testshuchu1-testtarget1)./testtarget1))/nEstimatePointstest1;
nEstimatePointstest2=length(testshuchu2);   TeachVariancetest2 = var(testshuchu2);                                 
meanerrortest2 = sum((testshuchu2 - testtarget2).^2)/nEstimatePointstest2;  
NRMSE_test2 = (sqrt(meanerrortest2./TeachVariancetest2));           
RMSE_test2=sqrt(meanerrortest2);
MAPE_test2 = sum(abs((testshuchu2-testtarget2)./testtarget2))/nEstimatePointstest2;
t2=clock;  time=t2-t1;  Time=[Time;time]; 
rmse_train1 = [rmse_train1;RMSE_train1]; mape_train1 = [mape_train1;MAPE_train1];
rmse_test1=[rmse_test1;RMSE_test1];mape_test1=[mape_test1;MAPE_test1];
rmse_train2 = [rmse_train2;RMSE_train2]; mape_train2 = [mape_train2;MAPE_train2];
rmse_test2=[rmse_test2;RMSE_test2];mape_test2=[mape_test2;MAPE_test2];
end
[mean(rmse_train1),mean(mape_train1),mean(rmse_test1),mean(mape_test1),mean(Time)]
[mean(rmse_train2),mean(mape_train2),mean(rmse_test2),mean(mape_test2)]
