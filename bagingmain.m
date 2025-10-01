%% preprocess
clc;
clear all;
WEKA_HOME = 'C:\Program Files\Weka-3-7';
javaaddpath([WEKA_HOME '\weka.jar']);
string=([WEKA_HOME '\data\breast.arff']);
   type=[0 0 0 0 0 1 0 0 0 0]; %% breast
%string=([WEKA_HOME '\data\mammographic.arff']);
  %type=[1 1 1 1 1 0]; %% mammographic
% string=([WEKA_HOME '\data\automobile.arff']);
%   type=[1 0 0 0 0 0 0 0 1 1 1 1 1 0 0 1 0 1 1 1 1 1 1 1 1 0];   %% automobile
%string=([WEKA_HOME '\data\cleveland.arff']);
  %%type=[1 0 0 1 0 0 0 1 0 1 0 0 0 0];   %% cleveland
loader = weka.core.converters.ArffLoader();
loader.setFile( java.io.File(string) );
dataset = loader.getDataSet();
dataset.setClassIndex(dataset.numAttributes()-1 );

attribute=dataset.numAttributes;
instance=dataset.numInstances;

%% substitute missing values 
for j=1:attribute
    datamod(:,j)=dataset.attributeToDoubleArray(j-1);
    datamod(:,j)=ModifyMissing(datamod(:,j),type(j));
end

%% split train & test
indexrand=randperm(instance);
b=round(.8*instance);
indextrain=indexrand(1:b);
indextest=indexrand(b+1:end);
train=datamod(indextrain,:);
test=datamod(indextest,:);
%% changing to arff
% prepare arff train
save train.txt train -ascii;
loader = weka.core.converters.MatlabLoader();
loader.setFile( java.io.File('train.txt') );
train = loader.getDataSet();
train.setClassIndex(train.numAttributes()-1 );
attribnotrain=train.numAttributes;
InstTrain=train.numInstances;
% Convert last attribute (class) from numeric to nominal
filter = weka.filters.unsupervised.attribute.NumericToNominal();
filter.setOptions( weka.core.Utils.splitOptions('-R last') );
filter.setInputFormat(train);   
train = filter.useFilter(train, filter);

%% prepare arff test
save test.txt test -ascii;
loader = weka.core.converters.MatlabLoader();
loader.setFile( java.io.File('test.txt') );
test = loader.getDataSet();
test.setClassIndex(test.numAttributes()-1 );

AttTest=test.numAttributes;
numinstancetest=test.numInstances;
% Convert last attribute (class) from numeric to nominal
filter = weka.filters.unsupervised.attribute.NumericToNominal();
filter.setOptions( weka.core.Utils.splitOptions('-R last') );
filter.setInputFormat(test);   
test = filter.useFilter(test, filter);

%% bagging

ensemble={};
data=train;

for i=1:21
    randInd=randsample(InstTrain,InstTrain,true);
    for j=1:length(randInd)
        data.add(train.instance(randInd(j)-1));
    end
    for j=1:length(randInd)
        data.delete(j-1);
    end
     % Train a C4.5 tree
    classifier = weka.classifiers.trees.J48();
    classifier.buildClassifier(data);
    ensemble{i}=classifier;
end
%%  test
estimatedTestLabels = zeros(21,numinstancetest);
predProbs = zeros(21,numinstancetest, train.numClasses());

for i=1:21
    for k=1:numinstancetest
        temp = ensemble{i}.classifyInstance(test.instance(k-1));
        estimatedTestLabels(i,k) = str2num(char(test.classAttribute().value((temp))));
        predProbs(i,k,:) = ensemble{i}.distributionForInstance( test.instance(k-1) );
    end  
end
%% Taking Vote
for i=1:numinstancetest
    PredTest(i)= mode(estimatedTestLabels(:,i));
        
end
num=0;
%% Accuracy
for i=1:numinstancetest
  if (PredTest(i)==test.instance(i-1).classValue )
      num=num+1;
  end
end
accuracy_bagging=(num/numinstancetest)*100
%%% calculate Q statistic
for i=1:21
    for j=1:numinstancetest
        if (estimatedTestLabels(i,j)==test.instance(j-1).classValue )
            label_oracle(i,j)=1;
        else
            label_oracle(i,j)=0;
        end
    end
end

for i=1:21
    for j=i+1:21
        N00=0;N10=0;
        N01=0;N11=0;
        for k=1: numinstancetest
            if (label_oracle(i,k)==label_oracle(j,k))&&(label_oracle(i,k)==0)
                N00=N00+1;
            else if (label_oracle(i,k)==label_oracle(j,k))&&(label_oracle(i,k)==1)
                N11=N11+1;
                else if (label_oracle(i,k)~=label_oracle(j,k))&&(label_oracle(i,k)==0)
                        N01=N01+1;
                    else if (label_oracle(i,k)~=label_oracle(j,k))&&(label_oracle(i,k)==1)
                            N10=N10+1;
                        end
                    end
                end
            end
      
        end
            Q(i,j)=(N11*N00-N01*N10)/(N11*N00+N01*N10);
        end
    end