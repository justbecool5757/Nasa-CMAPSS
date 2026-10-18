# NASA C-MAPSS Engine Health Dashboard

A Streamlit app for predictive maintenance using NASA's C-MAPSS turbofan engine dataset.

## Features
- 🔍 Explore sensor degradation trajectories
- 📈 Linear Regression to predict Remaining Useful Life (RUL)
- 🏷️ Classification of engine health (Healthy / Warning / Critical)
- 📊 One-way ANOVA across health classes
- 🔮 Real-time prediction from manual sensor input

## How to Run on Replit

1. Go to replit.com → Create Repl → Python
2. Upload all files from this folder
3. Upload your data files: train_FD001.txt, test_FD001.txt, RUL_FD001.txt
4. In the Shell tab, run:
   ```
   pip install -r requirements.txt
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```
5. Click the web preview link that appears

## Dataset Files Needed
- train_FD001.txt (or FD002/003/004)
- test_FD001.txt
- RUL_FD001.txt

## Reference
Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).
Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.
PHM'08, Denver CO.

