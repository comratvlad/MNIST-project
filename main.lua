---------------------------------------------------------
-- Главный модуль программы
-- Состоит из загрузки обучающей выборки,
-- создания сверточной сети, ее обучения и
-- тестирования
---------------------------------------------------------

require 'torch'
require 'nn'
require 'MNIST'
require 'MNISTNet'

---------------------------------------------------------
-- Загрузим выборку, нормализуем
dataset = MNIST.loadTrainset()
dataset:normalizeInput(0,255)

-- Разделим на обучающую и тестовую
trainset = {}
testset = {}

-- 32768 = 2^15
for i=1,42000 do
  if i<=32768 then
    trainset[i] = dataset[i]
  else
    testset[i-32768] = dataset[i]
  end
end

function trainset:size() return 32768 end
---------------------------------------------------------
-- Создаем сеть
net = MNISTNet.createSimpleCNN()
---------------------------------------------------------
-- Обучаем и сохраняем
MNISTNet.learnSimpleNet(net, trainset)
torch.save('net/simpleCNN.t7', net)
-- или загружаем
--net = torch.load('net/simpleCNN.t7')
---------------------------------------------------------
-- Проверяем результат
res = MNISTNet.testNet(net, testset)
print('Результат: ', res)
---------------------------------------------------------