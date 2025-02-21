// /app/api/chat/route.ts
import { getGroupConfig } from '@/app/actions';
import { serverEnv } from '@/env/server';
import { xai } from '@ai-sdk/xai';
import { cerebras } from '@ai-sdk/cerebras';
import { anthropic } from '@ai-sdk/anthropic'
import { google } from '@ai-sdk/google';
import { groq } from '@ai-sdk/groq'
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
const openrouter = createOpenRouter({
    apiKey: serverEnv.OPENROUTER_API_KEY,
});

import CodeInterpreter from '@e2b/code-interpreter';
import FirecrawlApp from '@mendable/firecrawl-js';
import { tavily } from '@tavily/core';
import {
    convertToCoreMessages,
    smoothStream,
    streamText,
    tool,
    createDataStreamResponse,
    wrapLanguageModel,
    extractReasoningMiddleware,
    customProvider,
    generateObject
} from 'ai';
import Exa from 'exa-js';
import { z } from 'zod';

const scira = customProvider({
    languageModels: {
        'scira-default': openrouter('x-ai/grok-2-1212'),
        'scira-grok-vision': openrouter('x-ai/grok-2-vision-1212'),
        'scira-sonnet': openrouter('anthropic/claude-3.5-sonnet'),
        'scira-r1': wrapLanguageModel({
            model: openrouter('openai/o3-mini-high'),
            middleware: extractReasoningMiddleware({ tagName: 'think' })
        }),
        'scira-llama': openrouter('deepseek/deepseek-r1-distill-llama-70b'),
    }
})

// Allow streaming responses up to 120 seconds
export const maxDuration = 300;

interface XResult {
    id: string;
    url: string;
    title: string;
    author?: string;
    publishedDate?: string;
    text: string;
    highlights?: string[];
    tweetId: string;
}

interface MapboxFeature {
    id: string;
    name: string;
    formatted_address: string;
    geometry: {
        type: string;
        coordinates: number[];
    };
    feature_type: string;
    context: string;
    coordinates: number[];
    bbox: number[];
    source: string;
}

interface GoogleResult {
    place_id: string;
    formatted_address: string;
    geometry: {
        location: {
            lat: number;
            lng: number;
        };
        viewport: {
            northeast: {
                lat: number;
                lng: number;
            };
            southwest: {
                lat: number;
                lng: number;
            };
        };
    };
    types: string[];
    address_components: Array<{
        long_name: string;
        short_name: string;
        types: string[];
    }>;
}

interface VideoDetails {
    title?: string;
    author_name?: string;
    author_url?: string;
    thumbnail_url?: string;
    type?: string;
    provider_name?: string;
    provider_url?: string;
}

interface VideoResult {
    videoId: string;
    url: string;
    details?: VideoDetails;
    captions?: string;
    timestamps?: string[];
    views?: string;
    likes?: string;
    summary?: string;
}

function sanitizeUrl(url: string): string {
    return url.replace(/\s+/g, '%20');
}

async function isValidImageUrl(url: string): Promise<boolean> {
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000);

        const response = await fetch(url, {
            method: 'HEAD',
            signal: controller.signal,
        });

        clearTimeout(timeout);

        return response.ok && (response.headers.get('content-type')?.startsWith('image/') ?? false);
    } catch {
        return false;
    }
}


const extractDomain = (url: string): string => {
    const urlPattern = /^https?:\/\/([^/?#]+)(?:[/?#]|$)/i;
    return url.match(urlPattern)?.[1] || url;
};

const deduplicateByDomainAndUrl = <T extends { url: string }>(items: T[]): T[] => {
    const seenDomains = new Set<string>();
    const seenUrls = new Set<string>();

    return items.filter(item => {
        const domain = extractDomain(item.url);
        const isNewUrl = !seenUrls.has(item.url);
        const isNewDomain = !seenDomains.has(domain);

        if (isNewUrl && isNewDomain) {
            seenUrls.add(item.url);
            seenDomains.add(domain);
            return true;
        }
        return false;
    });
};

// Modify the POST function to use the new handler
export async function POST(req: Request) {
    const { messages, model, group } = await req.json();
    const { tools: activeTools, systemPrompt } = await getGroupConfig(group);

    console.log("Running with model: ", model.trim());

    return createDataStreamResponse({
        execute: async (dataStream) => {
            const result = streamText({
                model: scira.languageModel(model),
                maxSteps: 5,
                providerOptions: {
                    'groq': {
                        reasoning_format: group === "fun" ? "raw" : "parsed",
                    }
                },
                messages: convertToCoreMessages(messages),
                experimental_transform: smoothStream({
                    chunking: 'word',
                    delayInMs: 15,
                }),
                temperature: 0,
                experimental_activeTools: [...activeTools],
                system: systemPrompt,
                tools: {
                    web_search: tool({
                        description: 'Search the web for information with multiple queries, max results and search depth.',
                        parameters: z.object({
                            queries: z.array(z.string().describe('Array of search queries to look up on the web.')),
                            maxResults: z.array(
                                z.number().describe('Array of maximum number of results to return per query.').default(10),
                            ),
                            topics: z.array(
                                z.enum(['general', 'news']).describe('Array of topic types to search for.').default('general'),
                            ),
                            searchDepth: z.array(
                                z.enum(['basic', 'advanced']).describe('Array of search depths to use.').default('basic'),
                            ),
                            exclude_domains: z
                                .array(z.string())
                                .describe('A list of domains to exclude from all search results.')
                                .default([]),
                        }),
                        execute: async ({
                            queries,
                            maxResults,
                            topics,
                            searchDepth,
                            exclude_domains,
                        }: {
                            queries: string[];
                            maxResults: number[];
                            topics: ('general' | 'news')[];
                            searchDepth: ('basic' | 'advanced')[];
                            exclude_domains?: string[];
                        }) => {
                            const apiKey = serverEnv.TAVILY_API_KEY;
                            const tvly = tavily({ apiKey });
                            const includeImageDescriptions = true;

                            console.log('Queries:', queries);
                            console.log('Max Results:', maxResults);
                            console.log('Topics:', topics);
                            console.log('Search Depths:', searchDepth);
                            console.log('Exclude Domains:', exclude_domains);

                            // Execute searches in parallel
                            const searchPromises = queries.map(async (query, index) => {
                                const data = await tvly.search(query, {
                                    topic: topics[index] || topics[0] || 'general',
                                    days: topics[index] === 'news' ? 7 : undefined,
                                    maxResults: maxResults[index] || maxResults[0] || 10,
                                    searchDepth: searchDepth[index] || searchDepth[0] || 'basic',
                                    includeAnswer: true,
                                    includeImages: true,
                                    includeImageDescriptions: includeImageDescriptions,
                                    excludeDomains: exclude_domains,
                                });

                                // Add annotation for query completion
                                dataStream.writeMessageAnnotation({
                                    type: 'query_completion',
                                    data: {
                                        query,
                                        index,
                                        total: queries.length,
                                        status: 'completed',
                                        resultsCount: data.results.length,
                                        imagesCount: data.images.length
                                    }
                                });

                                return {
                                    query,
                                    results: deduplicateByDomainAndUrl(data.results).map((obj: any) => ({
                                        url: obj.url,
                                        title: obj.title,
                                        content: obj.content,
                                        raw_content: obj.raw_content,
                                        published_date: topics[index] === 'news' ? obj.published_date : undefined,
                                    })),
                                    images: includeImageDescriptions
                                        ? await Promise.all(
                                            deduplicateByDomainAndUrl(data.images).map(
                                                async ({ url, description }: { url: string; description?: string }) => {
                                                    const sanitizedUrl = sanitizeUrl(url);
                                                    const isValid = await isValidImageUrl(sanitizedUrl);
                                                    return isValid
                                                        ? {
                                                            url: sanitizedUrl,
                                                            description: description ?? '',
                                                        }
                                                        : null;
                                                },
                                            ),
                                        ).then((results) =>
                                            results.filter(
                                                (image): image is { url: string; description: string } =>
                                                    image !== null &&
                                                    typeof image === 'object' &&
                                                    typeof image.description === 'string' &&
                                                    image.description !== '',
                                            ),
                                        )
                                        : await Promise.all(
                                            deduplicateByDomainAndUrl(data.images).map(async ({ url }: { url: string }) => {
                                                const sanitizedUrl = sanitizeUrl(url);
                                                return (await isValidImageUrl(sanitizedUrl)) ? sanitizedUrl : null;
                                            }),
                                        ).then((results) => results.filter((url): url is string => url !== null)),
                                };
                            });

                            const searchResults = await Promise.all(searchPromises);

                            return {
                                searches: searchResults,
                            };
                        },
                    }),
                    academic_search: tool({
                        description: 'Search academic papers and research.',
                        parameters: z.object({
                            query: z.string().describe('The search query'),
                        }),
                        execute: async ({ query }: { query: string }) => {
                            try {
                                const exa = new Exa(serverEnv.EXA_API_KEY as string);

                                // Search academic papers with content summary
                                const result = await exa.searchAndContents(query, {
                                    type: 'auto',
                                    numResults: 20,
                                    category: 'research paper',
                                    summary: {
                                        query: 'Abstract of the Paper',
                                    },
                                });

                                // Process and clean results
                                const processedResults = result.results.reduce<typeof result.results>((acc, paper) => {
                                    // Skip if URL already exists or if no summary available
                                    if (acc.some((p) => p.url === paper.url) || !paper.summary) return acc;

                                    // Clean up summary (remove "Summary:" prefix if exists)
                                    const cleanSummary = paper.summary.replace(/^Summary:\s*/i, '');

                                    // Clean up title (remove [...] suffixes)
                                    const cleanTitle = paper.title?.replace(/\s\[.*?\]$/, '');

                                    acc.push({
                                        ...paper,
                                        title: cleanTitle || '',
                                        summary: cleanSummary,
                                    });

                                    return acc;
                                }, []);

                                // Take only the first 10 unique, valid results
                                const limitedResults = processedResults.slice(0, 10);

                                return {
                                    results: limitedResults,
                                };
                            } catch (error) {
                                console.error('Academic search error:', error);
                                throw error;
                            }
                        },
                    }),
                    youtube_search: tool({
                        description: 'Search YouTube videos using Exa AI and get detailed video information.',
                        parameters: z.object({
                            query: z.string().describe('The search query for YouTube videos'),
                            no_of_results: z.number().default(5).describe('The number of results to return'),
                        }),
                        execute: async ({ query, no_of_results }: { query: string; no_of_results: number }) => {
                            try {
                                const exa = new Exa(serverEnv.EXA_API_KEY as string);

                                // Simple search to get YouTube URLs only
                                const searchResult = await exa.search(query, {
                                    type: 'keyword',
                                    numResults: no_of_results,
                                    includeDomains: ['youtube.com'],
                                });

                                // Process results
                                const processedResults = await Promise.all(
                                    searchResult.results.map(async (result): Promise<VideoResult | null> => {
                                        const videoIdMatch = result.url.match(
                                            /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&?/]+)/,
                                        );
                                        const videoId = videoIdMatch?.[1];

                                        if (!videoId) return null;

                                        // Base result
                                        const baseResult: VideoResult = {
                                            videoId,
                                            url: result.url,
                                        };

                                        try {
                                            // Fetch detailed info from our endpoints
                                            const [detailsResponse, captionsResponse, timestampsResponse] = await Promise.all([
                                                fetch(`${serverEnv.YT_ENDPOINT}/video-data`, {
                                                    method: 'POST',
                                                    headers: {
                                                        'Content-Type': 'application/json',
                                                    },
                                                    body: JSON.stringify({
                                                        url: result.url,
                                                    }),
                                                }).then((res) => (res.ok ? res.json() : null)),
                                                fetch(`${serverEnv.YT_ENDPOINT}/video-captions`, {
                                                    method: 'POST',
                                                    headers: {
                                                        'Content-Type': 'application/json',
                                                    },
                                                    body: JSON.stringify({
                                                        url: result.url,
                                                    }),
                                                }).then((res) => (res.ok ? res.text() : null)),
                                                fetch(`${serverEnv.YT_ENDPOINT}/video-timestamps`, {
                                                    method: 'POST',
                                                    headers: {
                                                        'Content-Type': 'application/json',
                                                    },
                                                    body: JSON.stringify({
                                                        url: result.url,
                                                    }),
                                                }).then((res) => (res.ok ? res.json() : null)),
                                            ]);

                                            // Return combined data
                                            return {
                                                ...baseResult,
                                                details: detailsResponse || undefined,
                                                captions: captionsResponse || undefined,
                                                timestamps: timestampsResponse || undefined,
                                            };
                                        } catch (error) {
                                            console.error(`Error fetching details for video ${videoId}:`, error);
                                            return baseResult;
                                        }
                                    }),
                                );

                                // Filter out null results
                                const validResults = processedResults.filter(
                                    (result): result is VideoResult => result !== null,
                                );

                                return {
                                    results: validResults,
                                };
                            } catch (error) {
                                console.error('YouTube search error:', error);
                                throw error;
                            }
                        },
                    }),
                    retrieve: tool({
                        description: 'Retrieve the information from a URL using Firecrawl.',
                        parameters: z.object({
                            url: z.string().describe('The URL to retrieve the information from.'),
                        }),
                        execute: async ({ url }: { url: string }) => {
                            const app = new FirecrawlApp({
                                apiKey: serverEnv.FIRECRAWL_API_KEY,
                            });
                            try {
                                const content = await app.scrapeUrl(url);
                                if (!content.success || !content.metadata) {
                                    return { error: 'Failed to retrieve content' };
                                }

                                // Define schema for extracting missing content
                                const schema = z.object({
                                    title: z.string(),
                                    content: z.string(),
                                    description: z.string()
                                });

                                let title = content.metadata.title;
                                let description = content.metadata.description;
                                let extractedContent = content.markdown;

                                // If any content is missing, use extract to get it
                                if (!title || !description || !extractedContent) {
                                    const extractResult = await app.extract([url], {
                                        prompt: "Extract the page title, main content, and a brief description.",
                                        schema: schema
                                    });

                                    if (extractResult.success && extractResult.data) {
                                        title = title || extractResult.data.title;
                                        description = description || extractResult.data.description;
                                        extractedContent = extractedContent || extractResult.data.content;
                                    }
                                }

                                return {
                                    results: [
                                        {
                                            title: title || 'Untitled',
                                            content: extractedContent || '',
                                            url: content.metadata.sourceURL,
                                            description: description || '',
                                            language: content.metadata.language,
                                        },
                                    ],
                                };
                            } catch (error) {
                                console.error('Firecrawl API error:', error);
                                return { error: 'Failed to retrieve content' };
                            }
                        },
                    }),
                    reason_search: tool({
                        description: 'Perform a reasoned web search with multiple steps and sources.',
                        parameters: z.object({
                            topic: z.string().describe('The main topic or question to research'),
                            depth: z.enum(['basic', 'advanced']).describe('Search depth level').default('basic'),
                        }),
                        execute: async ({ topic, depth }: { topic: string; depth: 'basic' | 'advanced' }) => {
                            const apiKey = serverEnv.TAVILY_API_KEY;
                            const tvly = tavily({ apiKey });
                            const exa = new Exa(serverEnv.EXA_API_KEY as string);

                            // Send initial plan status update (without steps count and extra details)
                            dataStream.writeMessageAnnotation({
                                type: 'research_update',
                                data: {
                                    id: 'research-plan-initial', // unique id for the initial state
                                    type: 'plan',
                                    status: 'running',
                                    title: 'Research Plan',
                                    message: 'Creating research plan...',
                                    timestamp: Date.now(),
                                    overwrite: true
                                }
                            });

                            // Now generate the research plan
                            const { object: researchPlan } = await generateObject({
                                model: openrouter('x-ai/grok-2-1212'),
                                temperature: 0.5,
                                schema: z.object({
                                    search_queries: z.array(z.object({
                                        query: z.string(),
                                        rationale: z.string(),
                                        source: z.enum(['web', 'academic', 'both']),
                                        priority: z.number().min(1).max(5)
                                    })).max(12),
                                    required_analyses: z.array(z.object({
                                        type: z.string(),
                                        description: z.string(),
                                        importance: z.number().min(1).max(5)
                                    })).max(8)
                                }),
                                prompt: `Create a focused research plan for the topic: "${topic}". 
                                        Keep the plan concise but comprehensive, with:
                                        - 4-12 targeted search queries (each can use web, academic, or both sources)
                                        - 2-8 key analyses to perform
                                        - Prioritize the most important aspects to investigate
                                        
                                        Consider different angles and potential controversies, but maintain focus on the core aspects.
                                        Ensure the total number of steps (searches + analyses) does not exceed 20.`
                            });

                            // Generate IDs for all steps based on the plan
                            const generateStepIds = (plan: typeof researchPlan) => {
                                // Generate an array of search steps.
                                const searchSteps = plan.search_queries.flatMap((query, index) => {
                                    if (query.source === 'both') {
                                        return [
                                            { id: `search-web-${index}`, type: 'web', query },
                                            { id: `search-academic-${index}`, type: 'academic', query }
                                        ];
                                    }
                                    const searchType = query.source === 'academic' ? 'academic' : 'web';
                                    return [{ id: `search-${searchType}-${index}`, type: searchType, query }];
                                });

                                // Generate an array of analysis steps.
                                const analysisSteps = plan.required_analyses.map((analysis, index) => ({
                                    id: `analysis-${index}`,
                                    type: 'analysis',
                                    analysis
                                }));

                                return {
                                    planId: 'research-plan',
                                    searchSteps,
                                    analysisSteps
                                };
                            };

                            const stepIds = generateStepIds(researchPlan);
                            let completedSteps = 0;
                            const totalSteps = stepIds.searchSteps.length + stepIds.analysisSteps.length;

                            // Complete plan status
                            dataStream.writeMessageAnnotation({
                                type: 'research_update',
                                data: {
                                    id: stepIds.planId,
                                    type: 'plan',
                                    status: 'completed',
                                    title: 'Research Plan',
                                    plan: researchPlan,
                                    totalSteps: totalSteps,
                                    message: 'Research plan created',
                                    timestamp: Date.now(),
                                    overwrite: true
                                }
                            });

                            // Execute searches
                            const searchResults = [];
                            let searchIndex = 0;  // Add index tracker

                            for (const step of stepIds.searchSteps) {
                                // Send running annotation for this search step
                                dataStream.writeMessageAnnotation({
                                    type: 'research_update',
                                    data: {
                                        id: step.id,
                                        type: step.type,
                                        status: 'running',
                                        title: step.type === 'web'
                                            ? `Searching the web for "${step.query.query}"`
                                            : step.type === 'academic'
                                                ? `Searching academic papers for "${step.query.query}"`
                                                : `Analyzing ${step.query.query}`,
                                        query: step.query.query,
                                        message: `Searching ${step.query.source} sources...`,
                                        timestamp: Date.now()
                                    }
                                });

                                if (step.type === 'web') {
                                    const webResults = await tvly.search(step.query.query, {
                                        searchDepth: depth,
                                        includeAnswer: true,
                                        maxResults: Math.min(6 - step.query.priority, 10)
                                    });

                                    searchResults.push({
                                        type: 'web',
                                        query: step.query,
                                        results: webResults.results.map(r => ({
                                            source: 'web',
                                            title: r.title,
                                            url: r.url,
                                            content: r.content
                                        }))
                                    });
                                    completedSteps++;
                                } else if (step.type === 'academic') {
                                    const academicResults = await exa.searchAndContents(step.query.query, {
                                        type: 'auto',
                                        numResults: Math.min(6 - step.query.priority, 5),
                                        category: 'research paper',
                                        summary: true
                                    });

                                    searchResults.push({
                                        type: 'academic',
                                        query: step.query,
                                        results: academicResults.results.map(r => ({
                                            source: 'academic',
                                            title: r.title || '',
                                            url: r.url || '',
                                            content: r.summary || ''
                                        }))
                                    });
                                    completedSteps++;
                                }

                                // Send completed annotation for the search step
                                dataStream.writeMessageAnnotation({
                                    type: 'research_update',
                                    data: {
                                        id: step.id,
                                        type: step.type,
                                        status: 'completed',
                                        title: step.type === 'web'
                                            ? `Searched the web for "${step.query.query}"`
                                            : step.type === 'academic'
                                                ? `Searched academic papers for "${step.query.query}"`
                                                : `Analysis of ${step.query.query} complete`,
                                        query: step.query.query,
                                        results: searchResults[searchResults.length - 1].results.map(r => {
                                            return { ...r };
                                        }),
                                        message: `Found ${searchResults[searchResults.length - 1].results.length} results`,
                                        timestamp: Date.now(),
                                        overwrite: true
                                    }
                                });

                                searchIndex++;  // Increment index
                            }

                            // Perform analyses
                            let analysisIndex = 0;  // Add index tracker

                            for (const step of stepIds.analysisSteps) {
                                dataStream.writeMessageAnnotation({
                                    type: 'research_update',
                                    data: {
                                        id: step.id,
                                        type: 'analysis',
                                        status: 'running',
                                        title: `Analyzing ${step.analysis.type}`,
                                        analysisType: step.analysis.type,
                                        message: `Analyzing ${step.analysis.type}...`,
                                        timestamp: Date.now()
                                    }
                                });

                                const { object: analysisResult } = await generateObject({
                                    model: scira.languageModel("scira-gemini-2"),
                                    temperature: 0.5,
                                    schema: z.object({
                                        findings: z.array(z.object({
                                            insight: z.string(),
                                            evidence: z.array(z.string()),
                                            confidence: z.number().min(0).max(1)
                                        })),
                                        implications: z.array(z.string()),
                                        limitations: z.array(z.string())
                                    }),
                                    prompt: `Perform a ${step.analysis.type} analysis on the search results. ${step.analysis.description}
                                            Consider all sources and their reliability.
                                            Search results: ${JSON.stringify(searchResults)}`
                                });

                                dataStream.writeMessageAnnotation({
                                    type: 'research_update',
                                    data: {
                                        id: step.id,
                                        type: 'analysis',
                                        status: 'completed',
                                        title: `Analysis of ${step.analysis.type} complete`,
                                        analysisType: step.analysis.type,
                                        findings: analysisResult.findings,
                                        message: `Analysis complete`,
                                        timestamp: Date.now(),
                                        overwrite: true
                                    }
                                });

                                analysisIndex++;  // Increment index
                            }

                            // After all analyses are complete, send running state for gap analysis
                            dataStream.writeMessageAnnotation({
                                type: 'research_update',
                                data: {
                                    id: 'gap-analysis',
                                    type: 'analysis',
                                    status: 'running',
                                    title: 'Research Gaps and Limitations',
                                    analysisType: 'gaps',
                                    message: 'Analyzing research gaps and limitations...',
                                    timestamp: Date.now()
                                }
                            });

                            // After all analyses are complete, analyze limitations and gaps
                            const { object: gapAnalysis } = await generateObject({
                                model: openrouter('x-ai/grok-2-1212'),
                                temperature: 0,
                                schema: z.object({
                                    limitations: z.array(z.object({
                                        type: z.string(),
                                        description: z.string(),
                                        severity: z.number().min(2).max(10),
                                        potential_solutions: z.array(z.string())
                                    })),
                                    knowledge_gaps: z.array(z.object({
                                        topic: z.string(),
                                        reason: z.string(),
                                        additional_queries: z.array(z.string())
                                    })),
                                    recommended_followup: z.array(z.object({
                                        action: z.string(),
                                        rationale: z.string(),
                                        priority: z.number().min(2).max(10)
                                    }))
                                }),
                                prompt: `Analyze the research results and identify limitations, knowledge gaps, and recommended follow-up actions.
                                        Consider:
                                        - Quality and reliability of sources
                                        - Missing perspectives or data
                                        - Areas needing deeper investigation
                                        - Potential biases or conflicts
                                        - Severity should be between 2 and 10
                                        - Knowledge gaps should be between 2 and 10
                                        
                                        Research results: ${JSON.stringify(searchResults)}
                                        Analysis findings: ${JSON.stringify(stepIds.analysisSteps.map(step => ({
                                            type: step.analysis.type,
                                            description: step.analysis.description,
                                            importance: step.analysis.importance
                                        })))}`
                            });

                            // Send gap analysis update
                            dataStream.writeMessageAnnotation({
                                type: 'research_update',
                                data: {
                                    id: 'gap-analysis',
                                    type: 'analysis',
                                    status: 'completed',
                                    title: 'Research Gaps and Limitations',
                                    analysisType: 'gaps',
                                    findings: gapAnalysis.limitations.map(l => ({
                                        insight: l.description,
                                        evidence: l.potential_solutions,
                                        confidence: (6 - l.severity) / 5
                                    })),
                                    gaps: gapAnalysis.knowledge_gaps,
                                    recommendations: gapAnalysis.recommended_followup,
                                    message: `Identified ${gapAnalysis.limitations.length} limitations and ${gapAnalysis.knowledge_gaps.length} knowledge gaps`,
                                    timestamp: Date.now(),
                                    overwrite: true,
                                    completedSteps: completedSteps + 1,
                                    totalSteps: totalSteps + (depth === 'advanced' ? 2 : 1)
                                }
                            });

                            let synthesis;

                            // If there are significant gaps and depth is 'advanced', perform additional research
                            if (depth === 'advanced' && gapAnalysis.knowledge_gaps.length > 0) {
                                const additionalQueries = gapAnalysis.knowledge_gaps.flatMap(gap => 
                                    gap.additional_queries.map(query => ({
                                        query,
                                        rationale: gap.reason,
                                        source: 'both' as const,
                                        priority: 3
                                    }))
                                );

                                // Execute additional searches for gaps
                                for (const query of additionalQueries) {
                                    // Generate a unique ID for this gap search
                                    const gapSearchId = `gap-search-${searchIndex++}`;
                                    
                                    // Send running annotation for this gap search
                                    dataStream.writeMessageAnnotation({
                                        type: 'research_update',
                                        data: {
                                            id: gapSearchId,
                                            type: 'web',
                                            status: 'running',
                                            title: `Additional search for "${query.query}"`,
                                            query: query.query,
                                            message: `Searching to fill knowledge gap: ${query.rationale}`,
                                            timestamp: Date.now()
                                        }
                                    });

                                    // Execute web search
                                    const webResults = await tvly.search(query.query, {
                                        searchDepth: depth,
                                        includeAnswer: true,
                                        maxResults: 5
                                    });

                                    // Add to search results
                                    searchResults.push({
                                        type: 'web',
                                        query: {
                                            query: query.query,
                                            rationale: query.rationale,
                                            source: 'web',
                                            priority: query.priority
                                        },
                                        results: webResults.results.map(r => ({
                                            source: 'web',
                                            title: r.title,
                                            url: r.url,
                                            content: r.content
                                        }))
                                    });

                                    // Send completed annotation for web search
                                    dataStream.writeMessageAnnotation({
                                        type: 'research_update',
                                        data: {
                                            id: gapSearchId,
                                            type: 'web',
                                            status: 'completed',
                                            title: `Additional web search for "${query.query}"`,
                                            query: query.query,
                                            results: webResults.results.map(r => ({
                                                source: 'web',
                                                title: r.title,
                                                url: r.url,
                                                content: r.content
                                            })),
                                            message: `Found ${webResults.results.length} results`,
                                            timestamp: Date.now(),
                                            overwrite: true
                                        }
                                    });

                                    // For 'both' source type, also do academic search
                                    if (query.source === 'both') {
                                        const academicSearchId = `gap-search-academic-${searchIndex++}`;
                                        
                                        // Send running annotation for academic search
                                        dataStream.writeMessageAnnotation({
                                            type: 'research_update',
                                            data: {
                                                id: academicSearchId,
                                                type: 'academic',
                                                status: 'running',
                                                title: `Additional academic search for "${query.query}"`,
                                                query: query.query,
                                                message: `Searching academic sources to fill knowledge gap: ${query.rationale}`,
                                                timestamp: Date.now()
                                            }
                                        });

                                        // Execute academic search
                                        const academicResults = await exa.searchAndContents(query.query, {
                                            type: 'auto',
                                            numResults: 3,
                                            category: 'research paper',
                                            summary: true
                                        });

                                        // Add to search results
                                        searchResults.push({
                                            type: 'academic',
                                            query: {
                                                query: query.query,
                                                rationale: query.rationale,
                                                source: 'academic',
                                                priority: query.priority
                                            },
                                            results: academicResults.results.map(r => ({
                                                source: 'academic',
                                                title: r.title || '',
                                                url: r.url || '',
                                                content: r.summary || ''
                                            }))
                                        });

                                        // Send completed annotation for academic search
                                        dataStream.writeMessageAnnotation({
                                            type: 'research_update',
                                            data: {
                                                id: academicSearchId,
                                                type: 'academic',
                                                status: 'completed',
                                                title: `Additional academic search for "${query.query}"`,
                                                query: query.query,
                                                results: academicResults.results.map(r => ({
                                                    source: 'academic',
                                                    title: r.title || '',
                                                    url: r.url || '',
                                                    content: r.summary || ''
                                                })),
                                                message: `Found ${academicResults.results.length} academic sources`,
                                                timestamp: Date.now(),
                                                overwrite: true
                                            }
                                        });
                                    }

                                    completedSteps++; // Increment completed steps counter
                                }

                                // Send running state for final synthesis
                                dataStream.writeMessageAnnotation({
                                    type: 'research_update',
                                    data: {
                                        id: 'final-synthesis',
                                        type: 'analysis',
                                        status: 'running',
                                        title: 'Final Research Synthesis',
                                        analysisType: 'synthesis',
                                        message: 'Synthesizing all research findings...',
                                        timestamp: Date.now()
                                    }
                                });

                                // Perform final synthesis of all findings
                                const { object: finalSynthesis } = await generateObject({
                                    model: scira.languageModel("scira-gemini-2"),
                                    temperature: 0,
                                    schema: z.object({
                                        key_findings: z.array(z.object({
                                            finding: z.string(),
                                            confidence: z.number().min(0).max(1),
                                            supporting_evidence: z.array(z.string())
                                        })),
                                        remaining_uncertainties: z.array(z.string())
                                    }),
                                    prompt: `Synthesize all research findings, including gap analysis and follow-up research.
                                            Highlight key conclusions and remaining uncertainties.
                                            
                                            Original results: ${JSON.stringify(searchResults)}
                                            Gap analysis: ${JSON.stringify(gapAnalysis)}
                                            Additional findings: ${JSON.stringify(additionalQueries)}`
                                });

                                synthesis = finalSynthesis;

                                // Send final synthesis update
                                dataStream.writeMessageAnnotation({
                                    type: 'research_update',
                                    data: {
                                        id: 'final-synthesis',
                                        type: 'analysis',
                                        status: 'completed',
                                        title: 'Final Research Synthesis',
                                        analysisType: 'synthesis',
                                        findings: finalSynthesis.key_findings.map(f => ({
                                            insight: f.finding,
                                            evidence: f.supporting_evidence,
                                            confidence: f.confidence
                                        })),
                                        uncertainties: finalSynthesis.remaining_uncertainties,
                                        message: `Synthesized ${finalSynthesis.key_findings.length} key findings`,
                                        timestamp: Date.now(),
                                        overwrite: true,
                                        completedSteps: totalSteps + (depth === 'advanced' ? 2 : 1) - 1,
                                        totalSteps: totalSteps + (depth === 'advanced' ? 2 : 1)
                                    }
                                });
                            }

                            // Final progress update
                            const finalProgress = {
                                id: 'research-progress',
                                type: 'progress' as const,
                                status: 'completed' as const,
                                message: `Research complete`,
                                completedSteps: totalSteps + (depth === 'advanced' ? 2 : 1),
                                totalSteps: totalSteps + (depth === 'advanced' ? 2 : 1),
                                isComplete: true,
                                timestamp: Date.now()
                            };

                            dataStream.writeMessageAnnotation({
                                type: 'research_update',
                                data: {
                                    ...finalProgress,
                                    overwrite: true
                                }
                            });

                            return {
                                plan: researchPlan,
                                results: searchResults,
                                synthesis: synthesis
                            };
                        },
                    }),
                },
                onChunk(event) {
                    if (event.chunk.type === 'tool-call') {
                        console.log('Called Tool: ', event.chunk.toolName);
                    }
                },
                onStepFinish(event) {
                    if (event.warnings) {
                        console.log('Warnings: ', event.warnings);
                    }
                },
                onFinish(event) {
                    console.log('Fin reason: ', event.finishReason);
                    console.log('Steps ', event.steps);
                    console.log('Messages: ', event.response.messages);
                },
                onError(event) {
                    console.log('Error: ', event.error);
                },
            });

            result.consumeStream();

            return result.mergeIntoDataStream(dataStream, {
                sendReasoning: true
            });
        }
    })
}