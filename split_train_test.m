function [trainsequence,testsequence] = split_train_test(sequence, trainsample, testsample)
trainsequence = sequence(1:trainsample,:) ; 
testsequence  = sequence(trainsample+1:trainsample+testsample,:);
