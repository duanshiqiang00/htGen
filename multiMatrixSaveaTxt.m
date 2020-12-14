clc
clear
temp=[];
temp2=[];
Fs = 128; Ts = 1/Fs;T = 20;
t = 0:Ts:T; t =t(1:length(t)-1);
x1 = 10.*sin(2*pi*t); x1 = x1';
x2 = 10.*cos(5*pi*t); x2 = x2';

for i=-180:1:180
temp = [temp,i];
temp2= temp.*0.1;
end
path = 'E:\Desktop\test.txt';
fid = fopen(path,'wt');

for i=1:length(x1)
    fprintf(fid,'%f\t%f\t%f\n',t(i),x2(i),x1(i));
%     fprintf(fid,'%f\n',x1(i));

end
fclose(fid);
fclose all;
% 
% for i=1:length(temp)
%     fprintf(fid,'%f\t%f\n',temp(i),temp2(i));
%     fclose(fid);
% end
% fclose all;
% 
% 
% 
