function data = fun_remove_outlier(data)
    to_del = [];
    Z_score = (data - mean(data)) / std(data); 
    for i = 1:numel(Z_score)
        if (Z_score(i) > 3) || (Z_score(i) < -3)
            to_del = [to_del, i];
        end
    end
    for i = 1:numel(to_del)
        data(to_del(i)) = data(to_del(i) - 1);  
    end
end
