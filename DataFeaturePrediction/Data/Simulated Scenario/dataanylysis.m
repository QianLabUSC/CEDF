
clear all;close all;
work_space = 'C:\Users\efeff\Desktop\usc\mud_exploration_decision_making';
cd([work_space, '\Data\Simulated Scenario']);

[num, text, raw]=xlsread('results.xls') 

magic_number = num(:, 4);
magic_number(isnan(magic_number)) = 0;

Interval = num(:,5);
initial_converage = num(:, 8);

histogram(magic_number);
xlabel('magic number 0 represents no magic number');
ylabel('number of participants')
title('magic number distribution')
figure;
histogram(Interval);
xlabel('Interval, ');
ylabel('number of participants')
title('distribution distribution');
figure;
[sorted_interval, order] = sort(Interval);
sorted_initial_converage = initial_converage(order);
plot(sorted_interval, sorted_initial_converage);