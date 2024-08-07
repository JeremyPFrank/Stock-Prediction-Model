import React ,{ useState } from 'react'
import './home.css'
import richDuck from './public/images/duckSing.png' 

import axios from 'axios';

function Home() {
  const [ticker, setTicker] = useState(null);
  const [start, setStart] = useState(null);
  const [period, setPeriod] = useState(null);
  const [result, setResult] = useState(null);
  const [epoch, setEpoch] = useState(null);
  const [batch, setBatch] = useState(null);
  const [setSize, setSetSize] = useState(null);
  const [random, setRandom] = useState(null);
  const [done, setDone] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/predict', {
        ticker,
        start,
        period,
        epoch,
        batch,
        setSize,
        random
      });
      setResult(response.data);
      setDone(true);
    
    } catch (error) {
      setResult('An error occurred');
      console.error(error);
    }
  };

  

  return (
    <div>
    <div className="homePage">
        <h1>
          Select Parameters And Run The Model
        </h1>
        <form onSubmit={handleSubmit}>
          
         <br/>
        <div className="inputFields"> 
          <label for="ticker">Stock Ticker: </label>
          <input
              type="text"
              id = "ticker"
              required = "true"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              /*onChange={(e) => setTicker(e.target.value)}
              /*value={this.state.value}*/
              
          />
          <p> </p>
          <label for="year">Training Start Year: </label>
          <input
              type="number"
              id = "year"
              /*value={this.state.value}
              onChange={this.handleChange}*/
              required minLength="4" 
              maxLength="4"
              value={start}
              onChange={(e) => setStart(e.target.value)}
          />
          <p> </p>
          <label for="period">Analysis Period (Days): </label>
          <input
              type="number"
              id = "period"
              /*value={this.state.value}
              onChange={this.handleChange}*/
              required minLength="1" 
              maxLength="4"
              value={period}
              onChange={(e) => setPeriod(e.target.value)}
          />
          <p> </p>
          <label for="random">Random Seed: </label>
          <input
              type="number"
              id = "random"
              /*value={this.state.value}
              onChange={this.handleChange}*/
              value={random}
              onChange={(e) => setRandom(e.target.value)}
          />
          <p> </p>
          <label for="size">Training Set Size (0-1):</label>
          <input
              type="range"
              id = "size"
              /*value={this.state.value}
              onChange={this.handleChange}*/
              min = "0.01" 
              max ="0.99"
              step = "0.01"
              value={setSize}
              onChange={(e) => setSetSize(e.target.value)}
              list = "sizes"
          />
          <output>{setSize}</output>
          <datalist id="sizes">
            <option value="0.1"></option>
            <option value="0.25"></option>
            <option value="0.5"></option>
            <option value="0.75"></option>
            <option value="0.9"></option>
          </datalist>
          <p> </p>
          <label for="epochs">LTSM Epochs:</label>
          <input
              type="range"
              id = "epochs"
              min = '1'
              max = '256'
              value={epoch}
              onChange={(e) => setEpoch(e.target.value)}
              list = "epochsList"
          />
          <output>{epoch}</output>
          <datalist id="epochsList">
            <option value="8"></option>
            <option value="16"></option>
            <option value="32"></option>
            <option value="64"></option>
            <option value="128"></option>
          </datalist>
          <p> </p>
          <label for="batch">LTSM Batch Size:</label>
          <input
              type="range"
              id = "batch"
              /*value={this.state.value}
              onChange={this.handleChange}*/
              min = '1'
              max = '256'
              value={batch}
              onChange={(e) => setBatch(e.target.value)}
              list = "batches"
          />
          <output>{batch}</output>
          <datalist id="batches">
            <option value="8"></option>
            <option value="16"></option>
            <option value="32"></option>
            <option value="64"></option>
            <option value="128"></option>
          </datalist>
          <p> </p>
          <button type="submit">Run Model</button>
         </div>

         </form>
       
      
      
      {done && <div className = 'results'>
        <p>Linear Model MSE: {result.lin_mse} </p>
        <p>LSTM Model MSE: {result.lstm_mse} </p>
        <p>Linear Model Prediction: {result.lin_pred} </p>
        <p>LSTM Model Prediction: {result.lstm_pred} </p>
      </div>}
      
      
    </div>
    <img 
        src= {richDuck}
        alt="Rich Duck"/>
    </div>
  );
}

export default Home;
