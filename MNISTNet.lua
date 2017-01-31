---------------------------------------------------------
-- Модуль для конструирования и обучения сверточных сетей,
-- предназначенных для решения задачи классификации
-- цифровых изображений рукописных цифр MNIST
---------------------------------------------------------

require 'torch'
require 'csvigo'

MNISTNet = {}

MNISTNet.path_submission = 'submission/submission.csv'

-- Возвращает максимум и его идекс в arr (Tensor)
function MNISTNet.findMax(arr)
  
  local size = arr:size()[1]
  local max = arr[1]
  local imax = 1
  for i=2,size do
    if arr[i]>max then
      max = arr[i]
      imax = i
    end
  end
  
  return max, imax

end
---------------------------------------------------------

-- Создание простой сверточной сети для распознавания цифр
function MNISTNet.createSimpleCNN()
  
  local net = nn.Sequential()

  net:add(nn.SpatialConvolutionMM(1, 16, 3, 3))
  net:add(nn.ReLU(false))
  net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  net:add(nn.SpatialConvolutionMM(16, 32, 3, 3))
  net:add(nn.ReLU(false))
  net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  net:add(nn.View(32*5*5))
  net:add(nn.Linear(32*5*5, 256))
  net:add(nn.ReLU(false))
  net:add(nn.Linear(256, 10))
  
  return net
  
end
---------------------------------------------------------

-- Обучение сети модулем StochasticGradient
function MNISTNet.learnSimpleNet(net, trainset)
  
  local criterion = nn.MSECriterion()

  local trainer = nn.StochasticGradient(net, criterion)
  trainer.learningRate = 0.1
  trainer.maxIteration = 5 -- just do 5 epochs of training.

  trainer:train(trainset)
  
end
---------------------------------------------------------

-- Тестирование созданной сети
function MNISTNet.testNet(net, testset)
  
  local size = #testset
  local correct = 0
  for i=1,size do
    
    local output = net:forward(testset[i][1])
    local label = testset[i][2]
    
    _,iOut = MNISTNet.findMax(output) 
    _,iLab = MNISTNet.findMax(label) 
    
    if iOut == iLab then correct = correct + 1 end
    
  end
  
  return correct/size
  
end
---------------------------------------------------------

-- Формирование submission-файла, для отправки на Kaggle
function MNISTNet.makeSubmission(net, testset)
  
  local size = #testset
  local submission = {}
  submission[1] = {'ImageId', 'Label'}
  
  for i=1,size do
    
    local output = net:forward(testset[i])
    
    _,iOut = MNISTNet.findMax(output) 
    iOut = iOut - 1

    submission[i + 1] = {i, iOut}
    
  end
  
  csvigo.save{path = MNISTNet.path_submission, data = submission}
  
end
---------------------------------------------------------