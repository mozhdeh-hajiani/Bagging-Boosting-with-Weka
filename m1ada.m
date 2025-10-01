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
for j=1:instance  %% Change Labels to -1 & 1
    if (datamod(j,end) == 0)
        datamod(j,end) = -1;
    end
end
%% split train & test
indexrand=randperm(instance);
b=round(.8*instance);
indextrain=indexrand(1:b);
indextest=indexrand(b+1:end);
train=datamod(indextrain,:);
test=datamod(indextest,:);
truelabeltrain = datamod(indextrain,end);
truelabeltest = datamod(indextest,end);
%% changing to arff
% train
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

%% test
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

%% ada boost.m1
 D = zeros(InstTrain,1)+(1/InstTrain);
beta = zeros(1,21);
 ensemble = {};
 randInst = train;  
 
 for i=1:21
    randInd = randsample(InstTrain,InstTrain,true,D);
    
    for j=1:length(randInd)
        randInst.add(train.instance(randInd(j)-1));
    end
    
    for j=1:length(randInd)
        randInst.delete(j-1);
    end
    % Train a C4.5 tree
    classifier = weka.classifiers.trees.J48();
    classifier.buildClassifier(randInst);
    
    ensemble{i} = classifier;
    
    %error
    for j=1:InstTrain
        temp = classifier.classifyInstance(train.instance(j-1));
        estimatedTrainLabels(j) = str2num(char(train.classAttribute().value((temp))));
       
    end
    error = 0;
    for j=1:InstTrain
        if estimatedTrainLabels(j) ~= truelabeltrain(j)
            error = error + D(j);
        end
    end
    if (error>=.5 || error==0)
        D(1:InstTrain,1)=1/InstTrain;
    else
        beta(i)=error/(1-error);
        if (beta(i)==0)
            beta(i)=.0001;
        end
        for j=1:InstTrain
            if (estimatedTrainLabels(j)~=truelabeltrain)
                L=1;
            else
                L=0;
            end
            D(j)=D(j)*(beta(i)^(1-L));
        end
        s=sum(D);
        D=D/s;
    end
 end
        
   
 %%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 %%%%%%test accuracy
estimatedTestLabels = zeros(21,numinstancetest);
predProbs = zeros(21,numinstancetest, train.numClasses());
for j=1:21
    for i=1:numinstancetest
        temp = ensemble{j}.classifyInstance(test.instance(i-1));
        estimatedTestLabels(j,i) = str2num(char(test.classAttribute().value((temp))));
     end
end
for i=1:numinstancetest
    index1 =find(estimatedTestLabels(:,i)==1);
    index2 =find(estimatedTestLabels(:,i)==-1);
    mu1 = sum(log(1./beta(index1)));
    mu2 = sum(log(1./beta(index2)));
    if(mu1>mu2)
         predictLabel(i)=1;
    else
         predictLabel(i)=-1;
    end
end
count = 0;
for i=1:numinstancetest
    if (predictLabel(i)==truelabeltest(i))

        count = count+1;
    end
end
accuracy_m1ada = (count/numinstancetest)*100
