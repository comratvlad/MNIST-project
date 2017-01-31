---------------------------------------------------------
-- Модуль, предназначенный для создания submission-файла,
-- с целью отправки его на Kaggle
---------------------------------------------------------

require 'torch'
require 'nn'
require 'MNIST'
require 'MNISTNet'

-- Загружаем сеть
local net = torch.load('net/simpleCNN.t7')
-- Загружаем тестовую выборку
local testset = MNIST.loadTestset(size) 
testset:normalizeInput(0, 255)
-- Формируем submission-файл
MNISTNet.makeSubmission(net, testset)