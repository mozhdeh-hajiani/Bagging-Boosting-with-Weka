function mD = ModifyMissing( D,datatype)
    index=[];
    k=0;
    if datatype==1   %%%%% If Numeric
        t=0;
        for i=1:length(D)
             if ~isnan(D(i))   
                t=t+D(i);
                k=k+1;
              
             else   
                 index=[index ;i];
             end
        end
        dmean=t/k;
        D(index)=dmean;
    else  %% If Nominal
        Category=unique(D);
        for i=1:length(D)
             if isnan(D(i))   
                 index=[index ;i];
             end
        end
        if (~isempty(index))
            for i=1:length(Category)
                nc(i)=length(find(D==Category(i)));
            end
            [ncMax indmax]=max(nc);
            D(index)=Category(indmax);
        end
    end
    mD=D;
end

