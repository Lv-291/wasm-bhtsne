/*
 * file from https://github.com/RReverser/wasm-bindgen-rayon
 */

import CopyPlugin from 'copy-webpack-plugin';
import { dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default {
    mode: 'production',
    entry: './index.js',
    output: {
        path: __dirname + '/dist',
        filename: 'index.js'
    },
    plugins: [
        new CopyPlugin({
            patterns: ['index.html']
        })
    ],
    module: {
        rules: [
            {
                test: /\.m?js$/,
                resolve: {
                    fullySpecified: false
                }
            }
        ]
    }
};