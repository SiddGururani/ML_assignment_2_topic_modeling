function [norm] = normalizeMatrix(M, dim)

[nr, nc] = size(M);

if(dim == 1)
    for i = 1:nr
        M(i,:) = M(i,:)./norm(M(i,:),1);
    end
else
    for i = 1:nc
        M(:,i) = M(:,i)./norm(M(:,i),1);
    end
end