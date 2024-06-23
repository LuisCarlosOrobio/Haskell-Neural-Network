{-# LANGUAGE PackageImports #-}

module Main where

import "random" System.Random (randomRIO) -- Importing random number generation functions
import Control.Monad (replicateM)         -- Importing replicateM to replicate monadic actions
import Data.List (transpose)              -- Importing transpose function to transpose matrices

-- Activation functions and their derivatives
sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x)) -- Sigmoid activation function

sigmoidDerivative :: Double -> Double
sigmoidDerivative x = x * (1 - x) -- Derivative of the sigmoid function

relu :: Double -> Double
relu x = max 0 x -- ReLU activation function

reluDerivative :: Double -> Double
reluDerivative x = if x > 0 then 1 else 0 -- Derivative of the ReLU function

-- Initialize weights with random values
initializeWeights :: Int -> Int -> IO [[Double]]
initializeWeights rows cols = replicateM rows (replicateM cols (randomRIO (-1.0, 1.0)))
-- Creates a matrix of specified size with random values between -1.0 and 1.0

-- Utility function to print weights
printWeights :: [[Double]] -> IO ()
printWeights = mapM_ (putStrLn . unwords . map show)
-- Prints each row of the weight matrix

-- Matrix multiplication
dotProduct :: [Double] -> [Double] -> Double
dotProduct xs ys = sum $ zipWith (*) xs ys -- Computes the dot product of two vectors

matrixVectorProduct :: [[Double]] -> [Double] -> [Double]
matrixVectorProduct mat vec = map (`dotProduct` vec) mat
-- Multiplies a matrix with a vector

-- Forward propagation
forwardPropagate :: [Double] -> [[Double]] -> (Double -> Double) -> [Double]
forwardPropagate input weights activationFunc = map activationFunc (matrixVectorProduct weights input)
-- Performs forward propagation through the network layer

-- Update weights with gradients
updateWeights :: [[Double]] -> [Double] -> [Double] -> Double -> [[Double]]
updateWeights weights errors inputs learningRate =
    zipWith (zipWith (\w (e, i) -> w + learningRate * e * i)) weights (map (\e -> zip (repeat e) inputs) errors)
-- Updates the weights using the calculated gradients

-- Backward propagation
backwardPropagate :: [Double] -> [Double] -> [Double] -> [[Double]] -> (Double -> Double) -> [Double] -> Double -> ([[Double]], [Double])
backwardPropagate input hiddenLayerOutput output outputLayerWeights activationDerivative target learningRate =
    let outputErrors = zipWith (-) target output -- Calculate the error at the output layer
        hiddenLayerErrors = map (sum . zipWith (*) outputErrors . map (sigmoidDerivative . reluDerivative)) (transpose outputLayerWeights)
        -- Calculate the error at the hidden layer
        outputLayerWeights' = updateWeights outputLayerWeights outputErrors hiddenLayerOutput learningRate
        -- Update the weights between hidden layer and output layer
    in (outputLayerWeights', hiddenLayerErrors)

-- Train the network
trainNetwork :: [[Double]] -> [[Double]] -> [Double] -> [Double] -> Double -> Int -> IO ([[Double]], [[Double]])
trainNetwork inputHiddenWeights hiddenOutputWeights input target learningRate epochs = go inputHiddenWeights hiddenOutputWeights epochs
  where
    go ihw how 0 = return (ihw, how) -- Base case: return the weights when epochs reach 0
    go ihw how n = do
        let hiddenLayerOutput = forwardPropagate input ihw relu -- Forward propagate through hidden layer
        let output = forwardPropagate hiddenLayerOutput how sigmoid -- Forward propagate through output layer
        let (how', hiddenLayerErrors) = backwardPropagate input hiddenLayerOutput output how sigmoidDerivative target learningRate
        -- Perform backward propagation to adjust weights
        let ihw' = updateWeights ihw hiddenLayerErrors input learningRate -- Update weights for hidden layer
        go ihw' how' (n - 1) -- Recursive call for the next epoch

main :: IO ()
main = do
    let input = [1.0, 0.5, -1.5] -- Input vector
    let target = [0.5] -- Target output
    inputHiddenWeights <- initializeWeights 3 4 -- Initialize weights between input and hidden layer
    hiddenOutputWeights <- initializeWeights 4 1 -- Initialize weights between hidden and output layer
    let learningRate = 0.1 -- Learning rate for training
    let epochs = 10000 -- Number of training epochs
    (ihw, how) <- trainNetwork inputHiddenWeights hiddenOutputWeights input target learningRate epochs
    -- Train the network and get the trained weights
    putStrLn "Trained Input-Hidden Weights:"
    printWeights ihw -- Print the trained input-hidden weights
    putStrLn "Trained Hidden-Output Weights:"
    printWeights how -- Print the trained hidden-output weights
    
    -- Verify the forward propagation with trained weights
    let hiddenLayerOutput = forwardPropagate input ihw relu -- Forward propagate through hidden layer
    let finalOutput = forwardPropagate hiddenLayerOutput how sigmoid -- Forward propagate through output layer
    putStrLn "Network Output after Training:"
    print finalOutput -- Print the network output after training
    putStrLn "Target Output:"
    print target -- Print the target output
