%% Convert Cell Array to Numeric Array  

%% 
% Convert numeric arrays in four cells of a cell array into one numeric array. 
C = {[1],    [2 3 4];
     [5; 9], [6 7 8; 10 11 12]}  

%%  
A = cell2mat(C)   

