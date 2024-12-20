// index.mjs
const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type,x-api-key',
    'Content-Type': 'application/json'
};

const createResponse = (statusCode, body) => ({
    statusCode,
    headers: corsHeaders,
    body: JSON.stringify(body)
});

const getRequiredFields = (tool) => {
    switch (tool) {
        case 'Enhancer':
            return 'copyText/uinput, crmChannel, brandVoice';
        case 'Evaluator':
            return 'uinput, crmChannel';
        case 'Writer':
            return 'campaignDescription, selectedCampaign, crmChannel, brandVoice';
        default:
            return 'unknown tool type';
    }
};

const validateToolInput = (body, tool) => {
    switch (tool) {
        case 'Evaluator':
            return !!body.uinput && !!body.crmChannel;
        case 'Enhancer':
            // Check for either copyText or uinput, plus other required fields
            return (!!body.copyText || !!body.uinput) && 
                   !!body.crmChannel && 
                   !!body.brandVoice && 
                   !!body.improvementContext;
        case 'Writer':
            return !!body.campaignDescription && 
                   !!body.selectedCampaign && 
                   !!body.crmChannel &&
                   !!body.brandVoice;
        default:
            return false;
    }
};

export const handler = async (event) => {
    // Handle preflight OPTIONS request
    if (event.httpMethod === 'OPTIONS') {
        return createResponse(200, {});
    }

    try {
        // Get and validate environment variables
        const { ACTUAL_LAMBDA_ENDPOINT, ACTUAL_LAMBDA_API_KEY } = process.env;
        
        if (!ACTUAL_LAMBDA_ENDPOINT || !ACTUAL_LAMBDA_API_KEY) {
            console.error('Missing required environment variables');
            throw new Error('Missing required environment variables');
        }

        // Validate incoming request body
        let body;
        try {
            body = JSON.parse(event.body || '{}');
            console.log('Received request body:', JSON.stringify(body));
            
            if (!body.tool) {
                return createResponse(400, {
                    error: 'Invalid request',
                    message: 'Tool type is required'
                });
            }

            if (!validateToolInput(body, body.tool)) {
                console.error('Invalid request - missing required fields:', JSON.stringify(body));
                return createResponse(400, {
                    error: 'Invalid request',
                    message: `Missing required fields. Required fields are: ${getRequiredFields(body.tool)}`
                });
            }
        } catch (e) {
            console.error('JSON Parse Error:', e);
            return createResponse(400, {
                error: 'Invalid JSON',
                message: 'Request body must be valid JSON'
            });
        }

        // If body has copyText but no uinput, map copyText to uinput
        if (body.copyText && !body.uinput) {
            body.uinput = body.copyText;
        }

        console.log('Making request to main Lambda');
        
        // Make the request to actual Lambda
        const response = await fetch(ACTUAL_LAMBDA_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-api-key': ACTUAL_LAMBDA_API_KEY
            },
            body: JSON.stringify(body)
        });

        console.log('Response status:', response.status);
        const data = await response.json();
        console.log('Response data:', JSON.stringify(data));

        if (!response.ok) {
            console.error('Lambda error response:', JSON.stringify(data));
            return createResponse(response.status, data);
        }

        return createResponse(200, data);

    } catch (error) {
        console.error('Error:', error);
        
        return createResponse(500, {
            error: 'Internal Server Error',
            message: 'An unexpected error occurred',
            details: error.message
        });
    }
};