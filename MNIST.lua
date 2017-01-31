---------------------------------------------------------
-- Модуль для работы с данными выборки MNIST
---------------------------------------------------------

require 'torch'
require 'csvigo'

MNIST = {}

MNIST.size_train = 42000
MNIST.size_test = 28000
MNIST.path_trainT7 = 'data/train.t7'
MNIST.path_testT7 = 'data/test.t7'
MNIST.path_trainCSV = 'data/train.csv'
MNIST.path_testCSV = 'data/test.csv'

-- Проверка существования файла по адресу filepath
function MNIST.fileExists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end
---------------------------------------------------------

-- Загрузка size элементов обучающей выборки из подготовленного *.asc файла
function MNIST.loadTrainset(size) 
  
  if size == nill or size >= MNIST.size_train then size = MNIST.size_train end
  
  if not MNIST.fileExists(MNIST.path_trainT7) then MNIST.createTrainsetFileT7() end
  
  local dataset = {}

  dataset = torch.load(MNIST.path_trainT7)
  
  function dataset:size() return size end
  
  function dataset:normalizeInput(mean, std)
    for i=1,size do
        dataset[i][1]:add(-mean)
        dataset[i][1]:mul(1/std)
    end
  end
  
  return dataset
  
end
---------------------------------------------------------

-- Загрузка size элементов тестовой выборки из подготовленного *.asc файла
function MNIST.loadTestset(size) 
  
  if size == nill or size >= MNIST.size_test then size = MNIST.size_test end
  
  if not MNIST.fileExists(MNIST.path_testT7) then MNIST.createTestsetFileT7() end
  
  local dataset = {}
  
  dataset = torch.load(MNIST.path_testT7)
  
  function dataset:normalizeInput(mean, std)
    for i=1,size do
        dataset[i]:add(-mean)
        dataset[i]:mul(1/std)
    end
  end
  
  function dataset:size() return size end
    
  return dataset
  
end
---------------------------------------------------------

-- Создание подготовленного *.t7 файла обучающей выборки
-- для дальнейшей быстрой загрузки
function MNIST.createTrainsetFileT7()
  
  local data = csvigo.load({path = MNIST.path_trainCSV, mode = 'large'})
  local size = MNIST.size_train
  
  local dataset = {}
  
  -- Идем по элементам выборки
  for i=2,size+1 do
    -- Текущая строка (элемент) выборки
    local sample = data[i]
    -- Формируем выход
    local output = torch.Tensor(10):zero()
    local labelIndex = sample[1] + 1
    output[labelIndex] = 1
    -- Формируем вход
    local input = torch.Tensor(1, 28, 28)
    table.remove(sample, 1)
    input[1] = torch.Tensor(sample):view(28,28)
    -- Формируем элемент dataset-а
    dataset[i - 1] = {input, output}
  end
  
  torch.save(MNIST.path_trainT7, dataset)
  
end
---------------------------------------------------------

-- Создание подготовленного *.t7 файла тестовой выборки
-- для дальнейшей быстрой загрузки
function MNIST.createTestsetFileT7()
  
  local data = csvigo.load({path = MNIST.path_testCSV, mode = 'large'})
  local size = MNIST.size_test
  
  local dataset = {}
  
  -- Идем по элементам выборки
  for i=2,size+1 do
    -- Текущая строка (элемент) выборки
    local sample = data[i]
    local input = torch.Tensor(1, 28, 28)
    input[1] = torch.Tensor(sample):view(28,28)
    -- Формируем элемент dataset-а
    dataset[i - 1] = input
  end
  
  torch.save(MNIST.path_testT7, dataset)
  
end
---------------------------------------------------------