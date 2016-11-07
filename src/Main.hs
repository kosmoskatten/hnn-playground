module Main where

import           AI.HNN.FF.Network
import           Control.Monad        (foldM, forM_)
import qualified Data.Vector.Storable as V
import           Text.Printf          (printf)

import           Mnist                (mnistFromFile)

main :: IO ()
main = mnistDigits

mnistDigits :: IO ()
mnistDigits = do
    Right samples <- mnistFromFile "datasets/mnist_train_10.csv"
    net           <- createNetwork 784 [30] 10
    trained       <- trainWithReport net 10000 20 samples
    testWithReport trained samples

simpleDigits :: IO ()
simpleDigits = do
    net     <- createNetwork 25 [10] 3
    trained <- trainWithReport net 10000 20 train
    testWithReport trained test

trainWithReport :: Network Float -> Int -> Int -> Samples Float -> IO (Network Float)
trainWithReport net' epochs reports samples = do
    let bursts = epochSet epochs reports
    foldM (\net n -> do
                printf "==> Start burst of %d epochs ...\n" n
                let newnet = trainNTimes n 2 sigmoid sigmoid' net samples
                let err = quadError sigmoid newnet samples
                printf "==> Done. Error %f\n" err
                return newnet
          ) net' bursts

testWithReport :: Network Float -> Samples Float -> IO ()
testWithReport net samples =
    forM_ samples $ \sample -> do
        let out = output net sigmoid (fst sample)
        printf "::> Output %s\n" (show out)
        printf "::> Expect %s\n" (show $ snd sample)
        putStrLn "==="

epochSet :: Int -> Int -> [Int]
epochSet epochs 0 = [epochs]
epochSet epochs reports =
    let frac = epochs `div` reports
        rest = epochs `mod` reports
    in reports `replicate` frac ++ [rest]

train :: Samples Float
train =
    [ V.fromList
        [ 0, 0, 1, 0, 0
        , 0, 1, 0, 1, 0
        , 0, 1, 1, 1, 0
        , 0, 1, 0, 1, 0
        , 0, 1, 0, 1, 0
        ] --> V.fromList [1, 0, 0]
    , V.fromList
        [ 0, 1, 1, 1, 0
        , 1, 0, 0, 0, 1
        , 1, 1, 1, 1, 1
        , 1, 0, 0, 0, 1
        , 1, 0, 0, 0, 1
        ] --> V.fromList [1, 0, 0]
    , V.fromList
        [ 0, 1, 1, 1, 0
        , 1, 0, 0, 1, 0
        , 1, 1, 1, 1, 0
        , 1, 0, 0, 1, 0
        , 1, 0, 0, 1, 0
        ] --> V.fromList [1, 0, 0]
    , V.fromList
        [ 1, 1, 0, 0, 0
        , 1, 0, 1, 0, 0
        , 1, 1, 0, 0, 0
        , 1, 0, 1, 0, 0
        , 1, 1, 0, 0, 0
        ] --> V.fromList [0, 1, 0]
    , V.fromList
        [ 1, 1, 1, 0, 0
        , 1, 0, 1, 0, 0
        , 1, 1, 0, 0, 0
        , 1, 0, 1, 0, 0
        , 1, 1, 1, 0, 0
        ] --> V.fromList [0, 1, 0]
    , V.fromList
        [ 1, 1, 1, 1, 1
        , 1, 0, 0, 1, 0
        , 1, 1, 1, 0, 0
        , 1, 0, 0, 1, 0
        , 1, 1, 1, 1, 1
        ] --> V.fromList [0, 1, 0]
    , V.fromList
        [ 0, 0, 1, 1, 1
        , 0, 1, 0, 0, 0
        , 1, 0, 0, 0, 0
        , 0, 1, 0, 0, 0
        , 0, 0, 1, 1, 1
        ] --> V.fromList [0, 0, 1]
    , V.fromList
        [ 0, 1, 1, 1, 1
        , 1, 0, 0, 0, 0
        , 1, 0, 0, 0, 0
        , 1, 0, 0, 0, 0
        , 0, 1, 1, 1, 1
        ] --> V.fromList [0, 0, 1]
    , V.fromList
        [ 1, 1, 1, 1, 1
        , 1, 0, 0, 0, 0
        , 1, 0, 0, 0, 0
        , 1, 0, 0, 0, 0
        , 1, 1, 1, 1, 1
        ] --> V.fromList [0, 0, 1]
    ]

test :: Samples Float
test =
    [ V.fromList
        [ 0, 1, 1, 0, 0
        , 0, 1, 0, 1, 0
        , 0, 1, 1, 1, 0
        , 0, 1, 0, 1, 0
        , 0, 1, 0, 1, 0
        ] --> V.fromList [1, 0, 0]
    , V.fromList
        [ 1, 1, 1, 1, 0
        , 1, 0, 0, 0, 1
        , 1, 1, 1, 1, 1
        , 1, 0, 0, 1, 1
        , 1, 0, 0, 0, 1
        ] --> V.fromList [1, 0, 0]
    , V.fromList
        [ 0, 1, 1, 1, 0
        , 1, 0, 0, 1, 0
        , 1, 1, 1, 1, 0
        , 1, 0, 0, 1, 1
        , 1, 0, 0, 1, 1
        ] --> V.fromList [1, 0, 0]
    , V.fromList
        [ 1, 1, 0, 0, 0
        , 1, 1, 1, 0, 0
        , 1, 1, 0, 0, 0
        , 1, 0, 1, 0, 0
        , 1, 1, 0, 0, 0
        ] --> V.fromList [0, 1, 0]
    , V.fromList
        [ 1, 1, 1, 1, 0
        , 1, 0, 1, 0, 0
        , 1, 1, 0, 0, 0
        , 1, 0, 1, 0, 0
        , 1, 1, 1, 0, 0
        ] --> V.fromList [0, 1, 0]
    , V.fromList
        [ 1, 1, 1, 1, 1
        , 1, 0, 0, 1, 1
        , 1, 1, 1, 0, 0
        , 1, 0, 0, 1, 0
        , 1, 1, 1, 1, 1
        ] --> V.fromList [0, 1, 0]
    , V.fromList
        [ 0, 0, 1, 1, 1
        , 0, 1, 1, 0, 0
        , 1, 0, 0, 0, 0
        , 0, 1, 1, 0, 0
        , 0, 0, 1, 1, 1
        ] --> V.fromList [0, 0, 1]
    , V.fromList
        [ 0, 1, 1, 1, 1
        , 1, 1, 0, 0, 0
        , 1, 1, 0, 0, 0
        , 1, 1, 0, 0, 0
        , 0, 1, 1, 1, 1
        ] --> V.fromList [0, 0, 1]
    , V.fromList
        [ 1, 1, 1, 1, 1
        , 1, 0, 0, 0, 0
        , 1, 0, 0, 0, 0
        , 1, 1, 0, 0, 0
        , 1, 1, 1, 1, 1
        ] --> V.fromList [0, 0, 1]
    ]
