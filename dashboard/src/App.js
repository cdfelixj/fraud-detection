import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import AlertPanel from './components/AlertPanel';
import './App.css';

const App = () => {
    return (
        <Router>
            <div className="app">
                <header className="app-header">
                    <h1>Fraud Detection System</h1>
                    <nav>
                        <a href="/">Dashboard</a>
                        <a href="/alerts">Alerts</a>
                    </nav>
                </header>
                <main className="app-main">
                    <Switch>
                        <Route path="/" exact component={Dashboard} />
                        <Route path="/alerts" component={AlertPanel} />
                    </Switch>
                </main>
            </div>
        </Router>
    );
};

export default App;