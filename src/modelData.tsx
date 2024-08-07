/*import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ModelData: React.FC = () => {
    const [message,setMessage] = useState<String>('Loading...');

    useEffect(() => {
        axios.get('http://127.0.0.1:5000/api/predict')
            .then(response => {
                setMessage(response.data.message);
            })
            .catch(error => {
                console.error('There was an error fetching the data!', error);
                setMessage('Error Getting Data');
            });
    }, []);

    return (
        <div>
            <p> {message} </p>
        </div>
    )
};

export default ModelData;*/